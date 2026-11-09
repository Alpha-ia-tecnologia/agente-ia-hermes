"""Ferramentas da empresa servidas pelo próprio bridge (MCP streamable HTTP).

Antes cada sessão registrava um servidor MCP stdio chamado
"ferramentas_da_empresa" — mas o Hermes registra servidores por NOME, uma vez
por processo: a 2ª conversa da empresa herdava as ferramentas (e o
conversationId embutido nas URLs) da 1ª. Agora cada sessão recebe um token
próprio; o nome do servidor deriva do token, então cada sessão enxerga só as
suas ferramentas, e nenhum processo extra é criado (antes: um Python por
instância só para servir ferramentas).
"""

import hashlib
import hmac
import json
import secrets

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from .mcp_tools_server import build_server

TOKEN_HEADER = "x-aao-tools"
# segredo por processo: tokens imprevisíveis mesmo sabendo tenant/sessão
_SECRET = secrets.token_bytes(32)
_tool_sets: dict[str, tuple[str, list[dict]]] = {}  # token -> (tenant, tools)


def register_tools(tenant_id: str, session_id: str, tools: list[dict]) -> str:
    """Guarda as ferramentas da sessão; mesmo conjunto -> mesmo token (o
    Hermes reaproveita o registro em vez de abrir outra conexão a cada turno)."""
    raw = json.dumps([tenant_id, session_id, tools], sort_keys=True).encode()
    token = hmac.new(_SECRET, raw, hashlib.sha256).hexdigest()[:32]
    _tool_sets[token] = (tenant_id, tools)
    return token


def forget_tenant_tools(tenant_id: str) -> None:
    """Instância parou: os registros MCP do processo morreram junto."""
    for token in [t for t, (tenant, _) in _tool_sets.items() if tenant == tenant_id]:
        _tool_sets.pop(token, None)


def _current_tools() -> list[dict]:
    request = server.request_context.request
    token = request.headers.get(TOKEN_HEADER, "") if request is not None else ""
    entry = _tool_sets.get(token)
    return entry[1] if entry else []


server = build_server(_current_tools)
# stateless: cada requisição é independente; nada a limpar no bridge
session_manager = StreamableHTTPSessionManager(app=server, stateless=True, json_response=True)
