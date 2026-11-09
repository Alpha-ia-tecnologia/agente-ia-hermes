import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from .acp_client import AcpError
from .config import settings
from .hermes_manager import agent_prompt, manager, session_provider_spec
from .mcp_http import TOKEN_HEADER, register_tools, session_manager
from pydantic import BaseModel

from .models import BridgeEvent, ChatRequest, ProviderConfig, SandboxConfig, ToolDef

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Aviso de fila a cada N segundos — também mantém o stream vivo (o fetch da
# api aborta após 300 s sem bytes).
QUEUE_NOTICE_SECONDS = 10

# Um turno por vez em cada sessão (conversa:agente): dois envios seguidos na
# mesma conversa entram em fila em vez de disputar a mesma sessão no Hermes.
_session_locks: dict[tuple[str, str], asyncio.Lock] = {}


def _build_mcp_servers(tenant_id: str, session_id: str, tools: list[ToolDef]) -> list[dict]:
    """Ferramentas do cliente como servidor MCP HTTP do próprio bridge, por sessão."""
    if not tools:
        return []
    token = register_tools(tenant_id, session_id, [t.model_dump() for t in tools])
    return [
        {
            "type": "http",
            # nome por token: o Hermes registra por nome, uma vez por processo
            "name": f"aao_{token[:8]}",
            "url": f"{settings.self_url}/mcp/",
            "headers": [
                {"name": TOKEN_HEADER, "value": token},
                {"name": "x-bridge-key", "value": settings.bridge_api_key},
            ],
        }
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.start_reaper()
    async with session_manager.run():
        yield
    await manager.shutdown()


app = FastAPI(title="AAO Hermes Bridge", version="0.1.0", lifespan=lifespan)


def _key_ok(value: str) -> bool:
    return (value or "").strip() == (settings.bridge_api_key or "").strip()


def check_api_key(x_bridge_key: str = Header(default="")) -> None:
    if not _key_ok(x_bridge_key):
        raise HTTPException(status_code=401, detail="invalid bridge key")


async def _mcp_asgi(scope, receive, send):
    if scope["type"] == "http":
        headers = dict(scope.get("headers") or [])
        if not _key_ok(headers.get(b"x-bridge-key", b"").decode()):
            await send({"type": "http.response.start", "status": 401, "headers": []})
            await send({"type": "http.response.body", "body": b"invalid bridge key"})
            return
    await session_manager.handle_request(scope, receive, send)


app.mount("/mcp", _mcp_asgi)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _ndjson(event: BridgeEvent) -> str:
    return event.model_dump_json(exclude_none=True) + "\n"


def _translate_update(update: dict, agent_id: str) -> BridgeEvent | None:
    """Map an ACP session/update payload to a normalized BridgeEvent."""
    kind = update.get("sessionUpdate")
    if kind == "agent_message_chunk":
        content = update.get("content") or {}
        if content.get("type") == "text":
            return BridgeEvent(type="token_chunk", agent_id=agent_id, text=content.get("text"))
    elif kind == "agent_thought_chunk":
        content = update.get("content") or {}
        return BridgeEvent(type="thought_chunk", agent_id=agent_id, text=content.get("text"))
    elif kind == "tool_call":
        title = update.get("title") or update.get("toolCallId", "ferramenta")
        is_delegation = "subagent" in str(update.get("kind", "")) or "delegate" in title.lower()
        return BridgeEvent(
            type="agent_delegation" if is_delegation else "tool_call",
            agent_id=agent_id,
            text=title,
            data={"toolCallId": update.get("toolCallId"), "kind": update.get("kind")},
        )
    elif kind == "tool_call_update":
        return BridgeEvent(
            type="tool_call_update",
            agent_id=agent_id,
            data={"toolCallId": update.get("toolCallId"), "status": update.get("status")},
        )
    elif kind == "plan":
        return BridgeEvent(type="agent_status", agent_id=agent_id, data={"plan": update.get("entries")})
    return None


@app.post("/chat", dependencies=[Depends(check_api_key)])
async def chat(req: ChatRequest) -> StreamingResponse:
    inst = manager.get(req.tenant_id)
    acp_session = req.session_id or "default"
    lock_key = (req.tenant_id, acp_session)

    async def stream():
        answer_parts: list[str] = []
        admitted = False
        client = None
        session_lock = _session_locks.setdefault(lock_key, asyncio.Lock())
        ticket: asyncio.Future | None = None
        try:
            # 1) fila por ordem de chegada: um turno por sessão + capacidade
            # global/por instância (o bridge admite quem chegou antes)
            waited = 0
            async with session_lock:
                ticket = manager.enqueue(inst)
                while not ticket.done():
                    if waited >= settings.queue_timeout_seconds:
                        yield _ndjson(
                            BridgeEvent(
                                type="error",
                                agent_id=req.agent.agent_id,
                                text="Sistema com muitos acessos agora — tente de novo em instantes.",
                            )
                        )
                        return
                    position = manager.position(ticket)
                    yield _ndjson(
                        BridgeEvent(
                            type="agent_status",
                            agent_id=req.agent.agent_id,
                            text=f"Muitos acessos agora — sua mensagem é a {position}ª da fila e será respondida em instantes…",
                            data={"queued": True, "position": position, "waitedSeconds": waited},
                        )
                    )
                    try:
                        await asyncio.wait_for(asyncio.shield(ticket), QUEUE_NOTICE_SECONDS)
                    except asyncio.TimeoutError:
                        waited += QUEUE_NOTICE_SECONDS
                admitted = True

                # 2) instância no ar (config = provedor padrão da empresa) e sessão
                # retomada com as ferramentas e o provedor dela — agente com
                # provedor próprio ou chave reserva não reinicia o processo
                base_provider = req.tenant_provider or req.provider
                client = await manager.ensure_started(inst, base_provider, req.sandbox)
                mcp_servers = _build_mcp_servers(req.tenant_id, acp_session, req.tools)
                await client.resume_session(
                    session_id=acp_session,
                    cwd=inst.session_cwd,
                    mcp_servers=mcp_servers,
                    timeout=settings.session_new_timeout_seconds,
                    provider_spec=None
                    if req.provider == base_provider
                    else session_provider_spec(req.provider),
                    send_provider=True,
                )
                client.open_session_stream(acp_session)
                yield _ndjson(BridgeEvent(type="session_started", session_id=acp_session, agent_id=req.agent.agent_id))

                blocks: list[dict] = [{"type": "text", "text": req.message}]
                for path in req.file_paths:
                    blocks.append({"type": "resource_link", "uri": f"file://{path}", "name": path.rsplit("/", 1)[-1]})
                done = await client.prompt(
                    acp_session, blocks, system_prompt=agent_prompt(req.agent.name, req.agent.system_prompt)
                )

                # 3) só os eventos DESTA sessão
                async for msg in client.notifications(acp_session):
                    inst.last_used = time.monotonic()
                    if msg.get("method") == "session/update":
                        params = msg.get("params") or {}
                        event = _translate_update(params.get("update") or {}, req.agent.agent_id)
                        if event:
                            if event.type == "token_chunk" and event.text:
                                answer_parts.append(event.text)
                            yield _ndjson(event)
                    # só o "wake" garante que a fila já tem tudo do turno: ele é
                    # enfileirado depois do último chunk. Parar no done.done()
                    # cortava o fim da resposta quando o leitor estava adiantado.
                    if msg.get("method") == "_bridge/wake" and done.done():
                        break

                if not done.done():
                    await asyncio.wait_for(done, settings.prompt_timeout_seconds)
                resp = done.result() or {}
                if "error" in resp:
                    raise AcpError(f"session/prompt: {resp['error']}")
            usage = (resp.get("result") or {}).get("usage")
            if usage and usage.get("totalTokens"):
                yield _ndjson(
                    BridgeEvent(
                        type="usage",
                        session_id=acp_session,
                        agent_id=req.agent.agent_id,
                        data={
                            "promptTokens": usage.get("inputTokens", 0),
                            "completionTokens": usage.get("outputTokens", 0),
                            "totalTokens": usage.get("totalTokens", 0),
                            "model": req.provider.model,
                        },
                    )
                )
            yield _ndjson(
                BridgeEvent(
                    type="final_message",
                    session_id=acp_session,
                    agent_id=req.agent.agent_id,
                    text="".join(answer_parts),
                )
            )
        except (AcpError, asyncio.TimeoutError) as exc:
            logger.exception("chat failed for tenant %s", req.tenant_id)
            yield _ndjson(BridgeEvent(type="error", agent_id=req.agent.agent_id, text=str(exc)))
        except Exception as exc:  # surface unexpected failures to aao-api instead of dropping the stream
            logger.exception("unexpected chat failure for tenant %s", req.tenant_id)
            yield _ndjson(BridgeEvent(type="error", agent_id=req.agent.agent_id, text=f"internal: {exc}"))
        finally:
            if client is not None:
                client.close_session_stream(acp_session)
            if admitted:
                manager.release(inst)
            elif ticket is not None:
                # desistiu na fila (timeout ou cliente saiu); devolve a vaga se
                # ela chegou entre o último aviso e o encerramento
                manager.cancel(inst, ticket)
            # sem dono nem fila: descarta o lock (não acumula um por conversa)
            if not session_lock.locked() and not getattr(session_lock, "_waiters", None):
                _session_locks.pop(lock_key, None)

    return StreamingResponse(stream(), media_type="application/x-ndjson")


class WarmupRequest(BaseModel):
    provider: ProviderConfig
    # mantido por compatibilidade com a api; a identidade do agente vai por turno
    system_prompt: str = ""
    sandbox: SandboxConfig = SandboxConfig()


@app.post("/tenants/{tenant_id}/warmup", dependencies=[Depends(check_api_key)])
async def warmup(tenant_id: str, req: WarmupRequest) -> dict:
    """Sobe a instância Hermes do tenant em background para evitar cold start."""
    inst = manager.get(tenant_id)
    already = inst.running
    if not already and not manager.can_warm(inst):
        # sem vaga livre: não recicla a instância de outra empresa só para aquecer
        return {"warming": False, "ready": False}

    async def _warm():
        try:
            # warmup só evita cold start: nunca derruba uma instância viva,
            # mesmo que a config seja diferente da atual
            client = await manager.ensure_started(
                inst, req.provider, req.sandbox, allow_restart=False
            )
            # O init pesado do agente (plugins, auto-detect de vision) é lazy e só
            # roda no primeiro session/new — uma sessão descartável pré-paga esse custo.
            await client.new_session(
                cwd=inst.session_cwd,
                timeout=settings.session_new_timeout_seconds,
            )
        except (AcpError, asyncio.TimeoutError) as exc:
            # warmup é melhor esforço: um chat pode reiniciar a instância no
            # meio do aquecimento (troca de config) — inofensivo, sem traceback
            logger.info("warmup abortado para tenant %s (%s) — inofensivo", tenant_id, exc)
        except Exception:
            logger.exception("warmup failed for tenant %s", tenant_id)

    if not already:
        asyncio.get_event_loop().create_task(_warm())
    return {"warming": not already, "ready": already}


@app.post("/tenants/{tenant_id}/stop", dependencies=[Depends(check_api_key)])
async def stop_tenant(tenant_id: str) -> dict:
    inst = manager.get(tenant_id)
    await inst.stop()
    return {"stopped": True}


# ---- Assinaturas (device-code / PKCE por tenant) ---------------------------

from .subscription_auth import (  # noqa: E402 — coeso com os endpoints abaixo
    ALL_PROVIDERS,
    DEVICE_FLOW_PROVIDERS,
    auth_manager,
    run_helper,
)


class AnthropicCodeRequest(BaseModel):
    code: str


class ImportCredentialRequest(BaseModel):
    source: str  # claude-code | codex-cli | gh-token
    payload: dict


@app.post("/tenants/{tenant_id}/auth/{provider}/start", dependencies=[Depends(check_api_key)])
async def auth_start(tenant_id: str, provider: str) -> dict:
    if provider not in DEVICE_FLOW_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"provider sem device-code: {provider}")
    flow = await auth_manager.start_flow(tenant_id, provider)
    if flow.error:
        raise HTTPException(status_code=502, detail=flow.error)
    return {"status": flow.status, "verification": flow.verification}


@app.get("/tenants/{tenant_id}/auth/{provider}/flow", dependencies=[Depends(check_api_key)])
def auth_flow_status(tenant_id: str, provider: str) -> dict:
    flow = auth_manager.get(tenant_id, provider)
    if flow is None:
        return {"status": "none"}
    return {"status": flow.status, "verification": flow.verification, "error": flow.error}


@app.delete("/tenants/{tenant_id}/auth/{provider}/flow", dependencies=[Depends(check_api_key)])
async def auth_flow_cancel(tenant_id: str, provider: str) -> dict:
    await auth_manager.cancel_flow(tenant_id, provider)
    return {"cancelled": True}


@app.post("/tenants/{tenant_id}/auth/anthropic/url", dependencies=[Depends(check_api_key)])
async def auth_anthropic_url(tenant_id: str) -> dict:
    result = await run_helper(tenant_id, "anthropic-url")
    if "auth_url" not in result:
        raise HTTPException(status_code=502, detail=result.get("message", "falha ao gerar URL"))
    return result


@app.post("/tenants/{tenant_id}/auth/anthropic/exchange", dependencies=[Depends(check_api_key)])
async def auth_anthropic_exchange(tenant_id: str, req: AnthropicCodeRequest) -> dict:
    result = await run_helper(tenant_id, "anthropic-exchange", req.code)
    if result.get("event") != "done":
        raise HTTPException(status_code=400, detail=result.get("message", "troca do código falhou"))
    return {"connected": True}


@app.post("/tenants/{tenant_id}/auth/import", dependencies=[Depends(check_api_key)])
async def auth_import(tenant_id: str, req: ImportCredentialRequest) -> dict:
    result = await run_helper(
        tenant_id, "import", req.source, stdin_data=json.dumps(req.payload)
    )
    if result.get("event") != "done":
        raise HTTPException(status_code=400, detail=result.get("message", "import falhou"))
    return {"connected": True, "provider": result.get("provider")}


@app.get("/tenants/{tenant_id}/auth/status", dependencies=[Depends(check_api_key)])
async def auth_status(tenant_id: str) -> dict:
    result = await run_helper(tenant_id, "status")
    return result


@app.delete("/tenants/{tenant_id}/auth/{provider}", dependencies=[Depends(check_api_key)])
async def auth_remove(tenant_id: str, provider: str) -> dict:
    if provider not in ALL_PROVIDERS and provider != "copilot":
        raise HTTPException(status_code=400, detail=f"provider desconhecido: {provider}")
    result = await run_helper(tenant_id, "remove", provider)
    if result.get("event") != "done":
        raise HTTPException(status_code=400, detail=result.get("message", "remoção falhou"))
    # instância ativa pode estar com credencial em memória — derruba
    inst = manager.get(tenant_id)
    await inst.stop()
    return {"removed": True}


# ---- Aprendizados (skills nativas do Hermes, isoladas por tenant) ----------


def _parse_skill_md(path: Path) -> dict:
    """Extrai frontmatter + corpo de um SKILL.md (parse simples, sem yaml)."""
    text = path.read_text(errors="replace")
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            body = parts[2].strip()
            for line in parts[1].splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    meta[key.strip()] = value.strip().strip('"').strip("'")
    return {"meta": meta, "body": body}


def _skill_dirs(root: Path) -> list[Path]:
    """Diretórios de skill (skills/<nome> ou skills/<categoria>/<nome>)."""
    found: list[Path] = []
    if not root.is_dir():
        return found
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if (child / "SKILL.md").is_file():
            found.append(child)
        else:
            for sub in sorted(child.iterdir()):
                if sub.is_dir() and not sub.name.startswith(".") and (sub / "SKILL.md").is_file():
                    found.append(sub)
    return found


@app.get("/tenants/{tenant_id}/skills", dependencies=[Depends(check_api_key)])
def list_skills(tenant_id: str) -> dict:
    skills_root = settings.tenants_dir / tenant_id / "skills"

    def entry(d: Path, archived: bool) -> dict:
        parsed = _parse_skill_md(d / "SKILL.md")
        stat = (d / "SKILL.md").stat()
        return {
            "name": parsed["meta"].get("name") or d.name,
            "description": parsed["meta"].get("description", ""),
            "version": parsed["meta"].get("version", ""),
            "archived": archived,
            "updatedAt": int(stat.st_mtime),
            "content": parsed["body"][:20000],
        }

    active = [entry(d, False) for d in _skill_dirs(skills_root)]
    archived = [entry(d, True) for d in _skill_dirs(skills_root / ".archive")]
    return {"skills": active + archived}


@app.post(
    "/tenants/{tenant_id}/skills/{skill_name}/archive",
    dependencies=[Depends(check_api_key)],
)
def archive_skill(tenant_id: str, skill_name: str) -> dict:
    """Move a skill para skills/.archive (recuperável — nunca exclui)."""
    skills_root = settings.tenants_dir / tenant_id / "skills"
    target = next(
        (d for d in _skill_dirs(skills_root) if
         (_parse_skill_md(d / "SKILL.md")["meta"].get("name") or d.name) == skill_name),
        None,
    )
    if target is None:
        raise HTTPException(status_code=404, detail="skill não encontrada")
    archive_dir = skills_root / ".archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / target.name
    if dest.exists():
        dest = archive_dir / f"{target.name}-{int(time.time())}"
    target.rename(dest)
    return {"archived": True}
