"""Minimal ACP (Agent Client Protocol) client over stdio JSON-RPC.

Speaks to `python -m acp_adapter.entry` from hermes-agent. stdout of the
subprocess carries newline-delimited JSON-RPC; stderr carries logs.

Um processo atende várias sessões ao mesmo tempo (vários usuários da mesma
empresa). Por isso as notificações são roteadas por sessionId para uma fila
por sessão, e cada session/prompt tem o seu próprio futuro — uma fila única
misturava e perdia tokens entre chats simultâneos.
"""

import asyncio
import contextlib
import json
import logging
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)

_CLOSED = {"method": "_bridge/closed"}
_WAKE = {"method": "_bridge/wake"}


class AcpError(Exception):
    pass


class AcpClient:
    def __init__(self, process: asyncio.subprocess.Process):
        self.process = process
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        # req_id do session/prompt -> sessionId, para acordar a fila certa
        self._prompt_sessions: dict[int, str] = {}
        self._queues: dict[str, asyncio.Queue[dict]] = {}
        self._closed = False
        self._reader_task = asyncio.create_task(self._read_loop())

    def _queue(self, session_id: str) -> asyncio.Queue[dict]:
        q = self._queues.get(session_id)
        if q is None:
            q = self._queues[session_id] = asyncio.Queue()
            if self._closed:
                q.put_nowait(_CLOSED)
        return q

    async def _read_loop(self) -> None:
        assert self.process.stdout is not None
        try:
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("non-JSON line on ACP stdout: %r", line[:200])
                    continue
                if "id" in msg and ("result" in msg or "error" in msg):
                    fut = self._pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        fut.set_result(msg)
                    # O fim de um session/prompt pode chegar depois da última
                    # notificação: acorda o consumidor daquela sessão.
                    session_id = self._prompt_sessions.pop(msg["id"], None)
                    if session_id is not None and session_id in self._queues:
                        self._queues[session_id].put_nowait(_WAKE)
                elif "method" in msg and "id" in msg:
                    # Server-initiated request (e.g. permission prompt) —
                    # auto-approve nothing; respond with method_not_found so
                    # the agent falls back to its non-interactive default.
                    await self._send(
                        {
                            "jsonrpc": "2.0",
                            "id": msg["id"],
                            "error": {"code": -32601, "message": "not supported by bridge"},
                        }
                    )
                else:
                    session_id = (msg.get("params") or {}).get("sessionId")
                    # só entrega para sessões com turno em andamento; o resto
                    # (avisos de comandos/uso fora de turno) é descartado
                    if session_id in self._queues:
                        self._queues[session_id].put_nowait(msg)
        finally:
            self._closed = True
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(AcpError("ACP process closed"))
            for q in self._queues.values():
                q.put_nowait(_CLOSED)

    async def _send(self, msg: dict) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write((json.dumps(msg) + "\n").encode())
        await self.process.stdin.drain()

    def _new_request(self, method: str, params: dict) -> tuple[int, asyncio.Future, dict]:
        req_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        return req_id, fut, {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}

    async def request(self, method: str, params: dict, timeout: float = 60) -> Any:
        if self._closed:
            raise AcpError("ACP process closed")
        _, fut, msg = self._new_request(method, params)
        await self._send(msg)
        resp = await asyncio.wait_for(fut, timeout)
        if "error" in resp:
            raise AcpError(f"{method}: {resp['error']}")
        return resp.get("result")

    def open_session_stream(self, session_id: str) -> None:
        """Começa a receber notificações da sessão (descarta restos antigos)."""
        self._queues.pop(session_id, None)
        self._queue(session_id)

    def close_session_stream(self, session_id: str) -> None:
        self._queues.pop(session_id, None)

    async def notifications(self, session_id: str) -> AsyncIterator[dict]:
        q = self._queue(session_id)
        while True:
            msg = await q.get()
            if msg is _CLOSED:
                return
            yield msg

    async def initialize(self) -> Any:
        return await self.request(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}},
            },
        )

    async def new_session(
        self, cwd: str, mcp_servers: Optional[list[dict]] = None, timeout: float = 60
    ) -> str:
        result = await self.request(
            "session/new", {"cwd": cwd, "mcpServers": mcp_servers or []}, timeout=timeout
        )
        return result["sessionId"]

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: Optional[list[dict]] = None,
        timeout: float = 60,
        provider_spec: Optional[dict] = None,
        send_provider: bool = False,
    ) -> str:
        params: dict = {"sessionId": session_id, "cwd": cwd, "mcpServers": mcp_servers or []}
        if send_provider:
            # provedor da sessão (patch AAO): dict ou None = o do config
            params["_meta"] = {"aaoProvider": provider_spec}
        result = await self.request("session/resume", params, timeout=timeout)
        # ResumeSessionResponse do adapter não ecoa sessionId — a sessão
        # resumida (ou recriada) mantém o id enviado na requisição.
        return (result or {}).get("sessionId") or session_id

    async def prompt(
        self, session_id: str, blocks: list[dict], system_prompt: Optional[str] = None
    ) -> asyncio.Future:
        """Dispara session/prompt; o futuro devolvido resolve quando o turno acaba."""
        if self._closed:
            raise AcpError("ACP process closed")
        params: dict = {"sessionId": session_id, "prompt": blocks}
        if system_prompt is not None:
            # identidade do agente por sessão (patch AAO no acp_adapter)
            params["_meta"] = {"aaoSystemPrompt": system_prompt}
        req_id, fut, msg = self._new_request("session/prompt", params)
        self._prompt_sessions[req_id] = session_id
        await self._send(msg)
        return fut

    async def close(self) -> None:
        self._reader_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._reader_task
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 10)
            except asyncio.TimeoutError:
                self.process.kill()
