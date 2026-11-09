"""Per-tenant Hermes instance lifecycle.

One ACP subprocess per tenant, isolated via HERMES_HOME. Instances start
lazily on first chat and are recycled after idle_timeout_seconds.
"""

import asyncio
import json
import logging
import os
import shutil
import time
from collections import deque
from pathlib import Path

from .acp_client import AcpClient
from .config import settings
from .mcp_http import forget_tenant_tools
from .models import ProviderConfig, SandboxConfig

logger = logging.getLogger(__name__)

# Maps AAO provider ids to hermes cli-config provider + base_url + key env var.
PROVIDER_MAP = {
    "openai": {"provider": "custom", "base_url": "https://api.openai.com/v1", "env": "OPENAI_API_KEY"},
    "deepseek": {"provider": "deepseek", "base_url": None, "env": "DEEPSEEK_API_KEY"},
    "anthropic": {"provider": "anthropic", "base_url": None, "env": "ANTHROPIC_API_KEY"},
    "google": {"provider": "gemini", "base_url": None, "env": "GOOGLE_API_KEY"},
    "openrouter": {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "env": "OPENROUTER_API_KEY"},
    "custom": {"provider": "custom", "base_url": None, "env": "OPENAI_API_KEY"},
    # Assinaturas: sem env de chave — o credential pool do Hermes resolve os
    # tokens do auth store do tenant (auth.json / pool / COPILOT_GITHUB_TOKEN).
    "openai-codex": {"provider": "openai-codex", "base_url": None, "env": None},
    "anthropic-oauth": {"provider": "anthropic", "base_url": None, "env": None},
    "copilot": {"provider": "github-copilot", "base_url": None, "env": None},
    "nous": {"provider": "nous", "base_url": None, "env": None},
    "xai-oauth": {"provider": "xai-oauth", "base_url": None, "env": None},
}

# Chaves de .env gerenciadas pelo provision (o merge preserva as demais,
# ex.: COPILOT_GITHUB_TOKEN gravado pelo login de assinatura).
MANAGED_ENV_KEYS = {
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
}


class TenantInstance:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.client: AcpClient | None = None
        self.last_used = time.monotonic()
        self.lock = asyncio.Lock()
        self.home = settings.tenants_dir / tenant_id
        self.config_fingerprint: str | None = None
        self.session_cwd: str = str(self.home / "workspace")
        self.sandbox_mode: str = "workspace"
        self.sandboxed: bool = False
        # turnos de chat em andamento nesta instância — um restart por mudança
        # de config espera zerar em vez de matar o processo no meio do turno
        self.busy: int = 0

    def provision(self, provider: ProviderConfig, sandbox: SandboxConfig) -> None:
        """Write per-tenant HERMES_HOME config before (re)starting.

        Só o que vale para a empresa inteira entra aqui (provider, sandbox,
        guardrail). A identidade de cada agente vai por sessão em
        agent_prompt() — assim trocar de agente não reinicia o processo.
        """
        self.home.mkdir(parents=True, exist_ok=True)
        self.sandbox_mode = sandbox.mode
        mapping = PROVIDER_MAP[provider.provider]
        base_url = provider.base_url or mapping["base_url"]

        # Toolsets por modo de sandbox: no workspace (padrão) os agentes não têm
        # terminal; em custom/free ganham terminal para atuar no computador,
        # sempre sob a regra de nunca excluir nada. "skills" habilita a
        # autoaprendizagem nativa do Hermes (SKILL.md em HERMES_HOME/skills,
        # isolado por tenant).
        # sem "delegation": o subagente genérico do Hermes não tem identidade;
        # o Gerente aciona os especialistas reais pela ferramenta da API
        toolsets = ["file", "todo", "vision", "skills"]
        if sandbox.mode in ("custom", "free"):
            toolsets.append("terminal")

        workspace = self.home / "workspace"
        if sandbox.mode == "custom" and sandbox.path:
            self.session_cwd = sandbox.path
        else:
            self.session_cwd = str(workspace)

        toolsets_yaml = ", ".join(f'"{t}"' for t in toolsets)
        config_lines = [
            "platform_toolsets:",
            f"  acp: [{toolsets_yaml}]",
            "model:",
            f'  default: "{provider.model}"',
            f'  provider: "{mapping["provider"]}"',
        ]
        if provider.api_key:
            # O provider "custom" não tem env var fixa: a chave vai no config.
            # Assinaturas não têm api_key — o credential pool resolve os tokens.
            config_lines.append(f'  api_key: "{provider.api_key}"')
        if base_url:
            config_lines.append(f'  base_url: "{base_url}"')
        if provider.temperature is not None:
            config_lines.append(f"  temperature: {provider.temperature}")
        guardrail = (
            "Você é um agente de escritório do Alpha Agents Office, a serviço de uma "
            "empresa cliente. SOBRE SUA IDENTIDADE: apresente-se sempre pelo seu nome "
            "de agente e como parte do Alpha Agents Office. NUNCA se apresente como "
            "Claude Code, Claude, ChatGPT, Codex, Copilot, Grok ou qualquer nome de "
            "modelo/fornecedor — a tecnologia por trás é um detalhe interno que não "
            "deve aparecer nas respostas. Responda de forma direta e objetiva, em "
            "português, sem jargão técnico de programação. SOBRE ESTILO: tom "
            "corporativo e profissional; NÃO use emojis (no máximo um, e só se o "
            "usuário usar ou pedir); saudações e perguntas simples recebem "
            "respostas curtas, de uma a três frases; use listas só quando ajudarem. "
            "Use ferramentas apenas "
            "quando a tarefa pedir. REGRA INEGOCIÁVEL: você é um copiloto — nunca "
            "exclua, apague ou "
            "sobrescreva destrutivamente arquivos, dados ou registros, mesmo que "
            "solicitado; nesses casos, explique que exclusões devem ser feitas "
            "manualmente pelo usuário. SOBRE DELEGAÇÕES: quando acionar outros "
            "agentes/especialistas, o trabalho deles acontece DENTRO da sua "
            "resposta atual — aguarde o resultado e entregue-o completo na mesma "
            "resposta. NUNCA diga que especialistas 'estão trabalhando em "
            "segundo plano', que resultados 'aparecerão automaticamente' ou peça "
            "para o usuário 'aguardar': nada continua rodando depois que sua "
            "resposta termina. Só afirme que algo foi feito quando a ferramenta "
            "retornou o resultado de fato. SOBRE ARQUIVOS E BUSCAS: só afirme que "
            "um arquivo ou pasta 'não existe' DEPOIS de checar com as ferramentas "
            "de arquivos e receber um resultado vazio de verdade. Se a ferramenta "
            "falhar, não estiver disponível, ou a pasta não estiver entre as "
            "autorizadas pelo conector, diga isso honestamente ('não consigo "
            "acessar essa pasta' / 'ela não está compartilhada comigo') em vez de "
            "concluir que não existe ou inventar caminhos e conteúdos. Não "
            "confunda suas próprias habilidades internas com as ferramentas, "
            "pastas ou programas do computador do usuário — são coisas distintas. "
            "SOBRE CÓDIGO: sempre que você gerar ou editar código ou um arquivo "
            "(HTML, script, config, etc.), MOSTRE o conteúdo relevante na sua "
            "resposta dentro de um bloco de código formatado com crase tripla e a "
            "linguagem (por exemplo ```html ... ```), para o usuário ver o que foi "
            "produzido; não responda apenas 'pronto, criei o arquivo'. Se o "
            "arquivo for muito grande, mostre as partes principais. "
            "SOBRE APRENDIZAGEM: quando o usuário "
            "ensinar um procedimento, uma regra ou uma preferência da empresa que "
            "valerá para o futuro (não só para esta conversa), registre como "
            "habilidade via skill_manage, escrita em português e sem incluir "
            "senhas, chaves ou dados pessoais sensíveis. Aplique as habilidades "
            "já aprendidas quando forem relevantes."
        )
        if sandbox.mode == "custom" and sandbox.path:
            guardrail += (
                f" Sua área de trabalho autorizada é exclusivamente '{sandbox.path}' "
                "(e o seu workspace de uploads); não leia nem altere nada fora dela."
            )
        elif sandbox.mode == "workspace":
            guardrail += (
                " Atue somente no seu workspace: leia arquivos apenas quando o "
                "usuário enviar documentos e nunca explore o sistema por conta própria."
            )
        escaped = guardrail.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        config_lines += ["system_prompt:", f'  append: "{escaped}"']
        (self.home / "config.yaml").write_text("\n".join(config_lines) + "\n")

        env_lines: list[str] = []
        if mapping["env"] and provider.api_key:
            env_lines.append(f'{mapping["env"]}={provider.api_key}')
            if mapping["provider"] == "custom":
                # custom provider also reads the generic OpenAI-compatible key
                env_lines.append(f"OPENAI_API_KEY={provider.api_key}")
        env_file = self.home / ".env"
        # Merge: preserva linhas não gerenciadas pelo provision (ex.:
        # COPILOT_GITHUB_TOKEN gravado pelo login de assinatura).
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                key = line.split("=", 1)[0].strip()
                if line.strip() and key not in MANAGED_ENV_KEYS:
                    env_lines.append(line)
        env_file.write_text("\n".join(env_lines) + "\n" if env_lines else "")
        env_file.chmod(0o600)

        workspace = self.home / "workspace"
        workspace.mkdir(exist_ok=True)

        # SOUL.md é a identidade primária do Hermes (auto-semeada com
        # "You are Hermes Agent... by Nous Research") — sobrescreve com a
        # identidade do produto para o agente nunca se apresentar como
        # Hermes/Claude Code/modelo subjacente.
        (self.home / "SOUL.md").write_text(
            "Você é um agente de escritório do Alpha Agents Office (AAO), a "
            "plataforma de escritórios de agentes de IA da empresa cliente. "
            "Sua identidade é o nome de agente definido no seu prompt (ex.: "
            "Agente Gerente) e você faz parte do Alpha Agents Office — é assim "
            "que você se apresenta, sempre em português. Você NUNCA se "
            "apresenta como Hermes, Nous Research, Claude, Claude Code, "
            "ChatGPT, Codex, Copilot, Grok ou qualquer nome de modelo ou "
            "fornecedor: a tecnologia por trás é detalhe interno e não aparece "
            "nas respostas, nem quando perguntarem qual modelo você usa "
            "(responda que é um agente do Alpha Agents Office). Você ajuda com "
            "as tarefas da empresa: análises, documentos, arquivos, projetos e "
            "tarefas do escritório — não é um assistente de programação.\n"
        )

    def _use_hard_sandbox(self) -> bool:
        """Sandbox duro (bwrap) para workspace/custom; FREE roda sem confinamento."""
        if self.sandbox_mode == "free":
            return False
        if settings.hard_sandbox is False:
            return False
        has_bwrap = shutil.which("bwrap") is not None
        if settings.hard_sandbox is True and not has_bwrap:
            logger.warning("hard_sandbox habilitado mas bwrap não encontrado")
        return has_bwrap

    def _bwrap_prefix(self) -> list[str]:
        """Confinamento: sistema read-only; escrita só no HERMES_HOME, na pasta
        custom do tenant e nos caches do uv. Rede compartilhada (APIs LLM/api)."""
        home = Path.home()
        rw_paths = [self.home]
        if self.sandbox_mode == "custom" and self.session_cwd != str(self.home / "workspace"):
            rw_paths.append(Path(self.session_cwd))
        for cache in (home / ".cache" / "uv", home / ".local" / "share" / "uv"):
            if cache.exists():
                rw_paths.append(cache)
        cmd = [
            "bwrap",
            "--ro-bind", "/", "/",
            "--dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            # esconde os HERMES_HOME dos outros tenants (contêm .env com
            # chaves); o bind do próprio home logo abaixo re-expõe só o dele
            "--tmpfs", str(settings.tenants_dir),
        ]
        for p in rw_paths:
            cmd += ["--bind", str(p), str(p)]
        cmd += ["--unshare-pid", "--die-with-parent", "--"]
        return cmd

    async def start(self) -> AcpClient:
        hermes_cmd = [
            "uv",
            "run",
            "--project",
            str(settings.hermes_repo_path),
            "python",
            "-m",
            "acp_adapter.entry",
        ]
        for sandboxed in ([True, False] if self._use_hard_sandbox() else [False]):
            cmd = (self._bwrap_prefix() + hermes_cmd) if sandboxed else hermes_cmd
            env = os.environ.copy()
            # o bridge roda sob `uv run` e exporta VIRTUAL_ENV do PRÓPRIO venv;
            # herdado pelo spawn do Hermes, o uv ignora com warning — remove
            # para o `--project` resolver o venv do hermes-agent sem ruído
            env.pop("VIRTUAL_ENV", None)
            env["HERMES_HOME"] = str(self.home)
            env["PYTHONUNBUFFERED"] = "1"
            # sessões não fixam provider: config atual (BYOK/assinatura/fallback)
            # sempre manda — ver patch em acp_adapter/session.py
            env["HERMES_ACP_PIN_SESSION_PROVIDER"] = "0"
            # turnos simultâneos neste processo (patch AAO no acp_adapter)
            env["HERMES_ACP_MAX_WORKERS"] = str(settings.max_turns_per_instance)
            # sessões vivas em memória (o resto fica no SessionDB e volta sob demanda)
            env["HERMES_ACP_MAX_LIVE_SESSIONS"] = str(settings.max_live_sessions)
            # ferramenta da empresa travada não prende o turno pelos 300 s padrão
            env["HERMES_ACP_MCP_TOOL_TIMEOUT"] = str(settings.mcp_tool_timeout_seconds)
            env["TERMINAL_CWD"] = self.session_cwd
            stderr_log = open(self.home / "hermes.stderr.log", "ab")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                # NÃO o repo do Hermes: os arquivos de contexto (AGENTS.md etc.)
                # caem no cwd do processo, e o guia de desenvolvimento do Hermes
                # (~30 KB) ia no prompt de todo agente de cliente. O pacote é
                # instalado no venv (uv --project), não depende do cwd.
                cwd=str(self.home),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr_log,
            )
            stderr_log.close()
            client = AcpClient(process)
            try:
                await client.initialize()
            except Exception:
                await client.close()
                # o stderr do Hermes vai para um arquivo — sem isto o log do
                # bridge só mostra "ACP process closed", sem o motivo real
                tail = self._stderr_tail()
                if tail:
                    logger.error(
                        "hermes stderr (tenant %s):\n%s", self.tenant_id, tail
                    )
                if sandboxed:
                    # bwrap indisponível de fato (ex.: user namespaces off):
                    # degrada para spawn direto em vez de derrubar o chat.
                    logger.warning(
                        "start confinado falhou para tenant %s — tentando sem bwrap",
                        self.tenant_id,
                    )
                    continue
                raise
            self.client = client
            self.sandboxed = sandboxed
            logger.info(
                "hermes instance started for tenant %s (pid %s, sandbox %s)",
                self.tenant_id,
                process.pid,
                "bwrap" if sandboxed else "off",
            )
            # vigia: se o processo morrer sozinho (não pelo nosso stop), loga o
            # motivo real (stderr) — crashes fora do start ficavam mudos
            asyncio.get_event_loop().create_task(self._watch_exit(process))
            return client
        raise RuntimeError("não foi possível iniciar a instância Hermes")

    def _stderr_tail(self, limit: int = 2000) -> str:
        try:
            return (self.home / "hermes.stderr.log").read_text(errors="replace")[-limit:]
        except OSError:
            return ""

    async def _watch_exit(self, process: asyncio.subprocess.Process) -> None:
        rc = await process.wait()
        # 0 = saída limpa; -15/143 = SIGTERM (nosso stop/restart/reciclagem —
        # 143 é o `uv run` repassando o sinal) — só crash importa
        if rc not in (0, -15, 143):
            tail = self._stderr_tail()
            logger.error(
                "hermes (tenant %s, pid %s) morreu com código %s. stderr:\n%s",
                self.tenant_id,
                process.pid,
                rc,
                tail or "(vazio)",
            )

    async def stop(self) -> None:
        client, self.client = self.client, None
        forget_tenant_tools(self.tenant_id)
        if client:
            await client.close()
            logger.info("hermes instance stopped for tenant %s", self.tenant_id)

    @property
    def running(self) -> bool:
        return self.client is not None and self.client.process.returncode is None


def session_provider_spec(provider: ProviderConfig) -> dict:
    """Provedor no formato do Hermes, para valer só numa sessão (_meta.aaoProvider)."""
    mapping = PROVIDER_MAP[provider.provider]
    return {
        "provider": mapping["provider"],
        "model": provider.model,
        "base_url": provider.base_url or mapping["base_url"],
        # assinaturas não têm chave: o credential pool do tenant resolve
        "api_key": provider.api_key or None,
    }


def agent_prompt(agent_name: str, system_prompt: str) -> str:
    """Identidade do agente, enviada por turno (prompt efêmero da sessão)."""
    parts = []
    if agent_name:
        # sem esta linha o modelo cai na identidade genérica ("sou o Alpha
        # Agents Office") quando o system_prompt não repete o próprio nome
        parts.append(
            f"Seu nome é {agent_name}. É assim que você se apresenta e é o "
            f"que responde quando perguntarem quem você é ou qual o seu "
            f"nome: você é {agent_name}, um agente do Alpha Agents Office."
        )
    if system_prompt:
        parts.append(system_prompt)
    return "\n\n".join(parts)


class HermesManager:
    """Instâncias por empresa + controle de entrada.

    Cada processo Hermes custa ~170 MB ociosos, então o número de processos
    vivos é limitado (max_instances): uma empresa nova recicla a instância
    ociosa usada há mais tempo; se todas estiverem em turno, o chat espera.
    Turnos também têm teto global e por instância. try_admit/release são
    síncronos (sem await) — atômicos no event loop, sem lock.
    """

    def __init__(self) -> None:
        self._instances: dict[str, TenantInstance] = {}
        self._reaper_task: asyncio.Task | None = None
        self.active_turns = 0
        self._waiters: deque[tuple[TenantInstance, asyncio.Future]] = deque()

    def get(self, tenant_id: str) -> TenantInstance:
        if tenant_id not in self._instances:
            self._instances[tenant_id] = TenantInstance(tenant_id)
        inst = self._instances[tenant_id]
        inst.last_used = time.monotonic()
        return inst

    @staticmethod
    def _occupies_slot(inst: TenantInstance) -> bool:
        # subindo (lock) ou com turno reservado também ocupa a vaga
        return inst.running or inst.busy > 0 or inst.lock.locked()

    def live_instances(self) -> int:
        return sum(1 for i in self._instances.values() if self._occupies_slot(i))

    def _evict_idle_lru(self, keep: TenantInstance) -> bool:
        idle = [
            i
            for i in self._instances.values()
            if i is not keep and i.running and i.busy == 0 and not i.lock.locked()
        ]
        if not idle:
            return False
        victim = min(idle, key=lambda i: i.last_used)
        # desliga o cliente já (síncrono) para a vaga valer neste mesmo tick;
        # o encerramento do processo segue em background
        client, victim.client = victim.client, None
        forget_tenant_tools(victim.tenant_id)
        logger.info("reciclando instância ociosa do tenant %s (limite de instâncias)", victim.tenant_id)
        if client:
            asyncio.get_event_loop().create_task(client.close())
        return True

    def try_admit(self, inst: TenantInstance) -> bool:
        """Reserva um turno se houver capacidade. O chamador DEVE chamar release()."""
        if self.active_turns >= settings.max_active_turns:
            return False
        if inst.busy >= settings.max_turns_per_instance:
            return False
        if not self._occupies_slot(inst) and self.live_instances() >= settings.max_instances:
            if not self._evict_idle_lru(keep=inst):
                return False
        self.active_turns += 1
        inst.busy += 1
        inst.last_used = time.monotonic()
        return True

    def enqueue(self, inst: TenantInstance) -> asyncio.Future:
        """Entra na fila por ordem de chegada; o futuro resolve quando o turno
        é admitido (o chamador DEVE chamar release() depois, ou cancel()).

        FIFO sem bloqueio de cabeça de fila: quem chegou antes é admitido
        primeiro, mas uma empresa no próprio teto não segura as outras.
        """
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._waiters.append((inst, fut))
        self._dispatch()
        return fut

    def position(self, fut: asyncio.Future) -> int:
        """Posição na fila (1 = próximo); 0 se já saiu dela."""
        for i, (_, f) in enumerate(self._waiters, start=1):
            if f is fut:
                return i
        return 0

    def cancel(self, inst: TenantInstance, fut: asyncio.Future) -> None:
        """Desistiu (timeout/cliente saiu). Se já tinha sido admitido, devolve a vaga."""
        self._waiters = deque(w for w in self._waiters if w[1] is not fut)
        if fut.done() and not fut.cancelled():
            self.release(inst)
        else:
            fut.cancel()

    def _dispatch(self) -> None:
        for waiter in list(self._waiters):
            if self.active_turns >= settings.max_active_turns:
                break
            inst, fut = waiter
            if fut.done():
                self._waiters.remove(waiter)
            elif self.try_admit(inst):
                self._waiters.remove(waiter)
                fut.set_result(True)

    def release(self, inst: TenantInstance) -> None:
        self.active_turns = max(0, self.active_turns - 1)
        inst.busy = max(0, inst.busy - 1)
        inst.last_used = time.monotonic()
        self._dispatch()

    def can_warm(self, inst: TenantInstance) -> bool:
        """Warmup só usa vaga livre — nunca recicla instância de outra empresa."""
        return self._occupies_slot(inst) or self.live_instances() < settings.max_instances

    async def ensure_started(
        self,
        inst: TenantInstance,
        provider: ProviderConfig,
        sandbox: SandboxConfig | None = None,
        *,
        allow_restart: bool = True,
    ) -> AcpClient:
        """Garante a instância do tenant no ar.

        Chamado depois de try_admit (o turno atual já conta em inst.busy).
        allow_restart=False (warmup) nunca derruba uma instância viva, mesmo
        com config diferente — warmup só existe para evitar cold start.
        """
        sandbox = sandbox or SandboxConfig()
        # sem nome/prompt do agente: eles vão por sessão, trocar de agente não
        # reinicia o processo
        fingerprint = json.dumps([provider.model_dump(), sandbox.model_dump()], sort_keys=True)
        async with inst.lock:
            stale = inst.running and inst.config_fingerprint != fingerprint
            if stale and allow_restart:
                # Config mudou (sandbox/provider): o Hermes só lê no start.
                # Espera os OUTROS turnos terminarem — matar o processo agora
                # derrubaria chats de outros usuários no meio.
                deadline = time.monotonic() + settings.restart_drain_timeout_seconds
                while inst.busy > 1 and time.monotonic() < deadline:
                    await asyncio.sleep(0.25)
                if inst.busy > 1:
                    logger.warning(
                        "tenant %s: %d turno(s) ainda ativos após drain — restart forçado",
                        inst.tenant_id,
                        inst.busy - 1,
                    )
                await inst.stop()
            if not inst.running:
                inst.provision(provider, sandbox)
                await inst.start()
                inst.config_fingerprint = fingerprint
            assert inst.client is not None
            return inst.client

    def start_reaper(self) -> None:
        self._reaper_task = asyncio.get_event_loop().create_task(self._reap_idle())

    async def _reap_idle(self) -> None:
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            for inst in list(self._instances.values()):
                if (
                    inst.running
                    and inst.busy == 0
                    and now - inst.last_used > settings.idle_timeout_seconds
                    and not inst.lock.locked()
                ):
                    await inst.stop()
                    self._dispatch()  # vaga liberada pode admitir quem espera

    async def shutdown(self) -> None:
        if self._reaper_task:
            self._reaper_task.cancel()
        for inst in self._instances.values():
            await inst.stop()


manager = HermesManager()
