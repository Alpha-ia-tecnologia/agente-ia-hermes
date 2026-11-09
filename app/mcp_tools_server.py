"""Servidor MCP (stdio) que expõe as ferramentas dinâmicas de um tenant.

Uso: python -m app.mcp_tools_server <caminho/tools.json>

O arquivo contém uma lista de ToolDef (ver app/models.py). O Hermes registra
este servidor via ACP (McpServerStdio) na criação da sessão; cada tools/call
vira uma requisição HTTP ao sistema do cliente. Erros nunca viram exceção —
retornam texto descritivo para o agente explicar ao usuário.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Callable

import httpx
import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

MAX_BODY_CHARS = 4000


def load_tools(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


async def execute_tool(tool: dict, args: dict) -> str:
    """Executa a chamada HTTP da ferramenta e devolve o resultado como texto."""
    method = tool.get("method", "POST").upper()
    timeout = tool.get("timeout_ms", 15000) / 1000
    headers = {"content-type": "application/json", **tool.get("headers", {})}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method == "GET":
                # merge com a query da URL: httpx substitui a query inteira
                # quando `params` é passado (perderia ?companyId=... fixo)
                url = httpx.URL(tool["url"]).copy_merge_params(args)
                resp = await client.get(url, headers=headers)
            else:
                resp = await client.request(method, tool["url"], json=args, headers=headers)
        body = resp.text[:MAX_BODY_CHARS]
        if resp.is_success:
            return f"Sucesso (HTTP {resp.status_code}):\n{body}"
        return (
            f"A API retornou erro HTTP {resp.status_code} ao chamar '{tool['name']}'. "
            f"Resposta: {body or '(vazia)'}"
        )
    except httpx.TimeoutException:
        return (
            f"A chamada à ferramenta '{tool['name']}' excedeu o tempo limite "
            f"de {int(timeout)}s. O sistema externo pode estar lento ou fora do ar."
        )
    except httpx.HTTPError as exc:
        return f"Falha de rede ao chamar a ferramenta '{tool['name']}': {exc}"


def build_server(tools: "list[dict] | Callable[[], list[dict]]") -> Server:
    """`tools` pode ser a lista fixa (stdio) ou uma função que a resolve por
    requisição (servidor HTTP do bridge, uma lista por sessão)."""
    get_tools = tools if callable(tools) else (lambda: tools)
    server = Server("aao-ferramentas")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=t["name"],
                description=t.get("description", ""),
                inputSchema=t.get("params_schema") or {"type": "object", "properties": {}},
            )
            for t in get_tools()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        tool = next((t for t in get_tools() if t["name"] == name), None)
        if tool is None:
            text = f"Ferramenta '{name}' não encontrada."
        else:
            text = await execute_tool(tool, arguments or {})
        return [types.TextContent(type="text", text=text)]

    return server


async def main() -> None:
    tools = load_tools(sys.argv[1])
    server = build_server(tools)
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
