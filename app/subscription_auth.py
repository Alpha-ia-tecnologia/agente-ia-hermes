"""Fluxos de login por assinatura (device-code/PKCE) por tenant.

Cada fluxo roda o hermes_auth_helper como subprocess com HERMES_HOME do
tenant; este módulo acompanha os eventos NDJSON e guarda o estado em memória
(1 fluxo ativo por tenant+provider — igual à restrição do conector).
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)

HELPER_PATH = Path(__file__).resolve().parent / "hermes_auth_helper.py"

DEVICE_FLOW_PROVIDERS = {"openai-codex", "nous", "xai-oauth", "copilot"}
ALL_PROVIDERS = DEVICE_FLOW_PROVIDERS | {"anthropic"}
FLOW_TIMEOUT_SECONDS = 15 * 60


def _helper_cmd(*args: str) -> list[str]:
    return [
        "uv",
        "run",
        "--project",
        str(settings.hermes_repo_path),
        "--directory",
        str(settings.hermes_repo_path),
        "python",
        str(HELPER_PATH),
        *args,
    ]


def _tenant_env(tenant_id: str) -> dict:
    import os

    env = os.environ.copy()
    home = settings.tenants_dir / tenant_id
    home.mkdir(parents=True, exist_ok=True)
    env["HERMES_HOME"] = str(home)
    env["PYTHONUNBUFFERED"] = "1"
    return env


async def run_helper(tenant_id: str, *args: str, stdin_data: str | None = None) -> dict:
    """Executa um comando síncrono do helper e retorna o último JSON emitido."""
    proc = await asyncio.create_subprocess_exec(
        *_helper_cmd(*args),
        env=_tenant_env(tenant_id),
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(
        proc.communicate(stdin_data.encode() if stdin_data is not None else None),
        timeout=120,
    )
    last: dict = {}
    for line in stdout.decode().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    if proc.returncode != 0 and not last:
        raise RuntimeError(stderr.decode()[-300:] or "auth helper falhou")
    return last


@dataclass
class AuthFlow:
    tenant_id: str
    provider: str
    process: asyncio.subprocess.Process
    started_at: float = field(default_factory=time.monotonic)
    verification: dict | None = None
    done: bool = False
    error: str | None = None
    _reader: asyncio.Task | None = None

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        if self.done:
            return "connected"
        if time.monotonic() - self.started_at > FLOW_TIMEOUT_SECONDS:
            return "expired"
        return "pending" if self.verification else "starting"


class SubscriptionAuthManager:
    def __init__(self) -> None:
        self._flows: dict[tuple[str, str], AuthFlow] = {}

    def get(self, tenant_id: str, provider: str) -> AuthFlow | None:
        return self._flows.get((tenant_id, provider))

    async def start_flow(self, tenant_id: str, provider: str) -> AuthFlow:
        if provider not in DEVICE_FLOW_PROVIDERS:
            raise ValueError(f"provider sem fluxo device-code: {provider}")
        existing = self._flows.get((tenant_id, provider))
        if existing and existing.status in ("starting", "pending"):
            return existing
        process = await asyncio.create_subprocess_exec(
            *_helper_cmd("flow", provider),
            env=_tenant_env(tenant_id),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        flow = AuthFlow(tenant_id=tenant_id, provider=provider, process=process)
        flow._reader = asyncio.get_event_loop().create_task(self._read_events(flow))
        self._flows[(tenant_id, provider)] = flow

        # espera a URL/código de verificação (ou erro) por até 45s
        for _ in range(90):
            if flow.verification or flow.error or flow.done:
                break
            await asyncio.sleep(0.5)
        return flow

    async def _read_events(self, flow: AuthFlow) -> None:
        assert flow.process.stdout is not None
        try:
            while True:
                line = await flow.process.stdout.readline()
                if not line:
                    break
                try:
                    event = json.loads(line.decode().strip())
                except json.JSONDecodeError:
                    continue
                kind = event.get("event")
                if kind == "verification":
                    flow.verification = {"uri": event.get("uri"), "code": event.get("code")}
                elif kind == "done":
                    flow.done = True
                elif kind == "error":
                    flow.error = event.get("message", "erro desconhecido")
            await flow.process.wait()
            if flow.process.returncode != 0 and not flow.error and not flow.done:
                stderr = b""
                if flow.process.stderr is not None:
                    stderr = await flow.process.stderr.read()
                flow.error = (stderr.decode()[-300:] or "fluxo encerrou sem sucesso").strip()
        except Exception as exc:
            flow.error = str(exc)
            logger.exception(
                "auth flow reader falhou (tenant %s, provider %s)",
                flow.tenant_id,
                flow.provider,
            )

    async def cancel_flow(self, tenant_id: str, provider: str) -> None:
        flow = self._flows.pop((tenant_id, provider), None)
        if flow and flow.process.returncode is None:
            flow.process.terminate()


auth_manager = SubscriptionAuthManager()
