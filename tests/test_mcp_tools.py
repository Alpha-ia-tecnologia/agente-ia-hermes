import asyncio
import json

import httpx
import pytest

from app.mcp_tools_server import build_server, execute_tool

TOOL = {
    "name": "cadastrar_usuario",
    "description": "Cadastra um usuário no sistema X",
    "method": "POST",
    "url": "http://sistema-x.local/api/usuarios",
    "headers": {"x-api-key": "abc"},
    "params_schema": {
        "type": "object",
        "properties": {"nome": {"type": "string"}},
        "required": ["nome"],
    },
    "timeout_ms": 2000,
}


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_execute_tool_success(monkeypatch):
    async def fake_request(self, method, url, **kwargs):
        assert method == "POST" and kwargs["json"] == {"nome": "Ana"}
        assert kwargs["headers"]["x-api-key"] == "abc"
        return httpx.Response(201, text='{"id": 1}')

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)
    result = _run(execute_tool(TOOL, {"nome": "Ana"}))
    assert "Sucesso (HTTP 201)" in result and '"id": 1' in result


def test_execute_tool_http_error(monkeypatch):
    async def fake_request(self, method, url, **kwargs):
        return httpx.Response(500, text="boom")

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)
    result = _run(execute_tool(TOOL, {}))
    assert "erro HTTP 500" in result and "cadastrar_usuario" in result


def test_execute_tool_timeout(monkeypatch):
    async def fake_request(self, method, url, **kwargs):
        raise httpx.ConnectTimeout("timeout")

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)
    result = _run(execute_tool(TOOL, {}))
    assert "tempo limite" in result


def test_execute_tool_get_uses_query_params(monkeypatch):
    captured = {}

    async def fake_get(self, url, **kwargs):
        captured["url"] = httpx.URL(url)
        captured.update(kwargs)
        return httpx.Response(200, text="ok")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    # query fixa na URL (ex.: ?companyId=) precisa sobreviver aos argumentos
    tool = {**TOOL, "method": "GET", "url": "http://sistema-x.local/api/usuarios?companyId=c1"}
    result = _run(execute_tool(tool, {"nome": "Ana"}))
    assert captured["url"].params["nome"] == "Ana"
    assert captured["url"].params["companyId"] == "c1"
    assert "params" not in captured
    assert "Sucesso" in result


def test_build_server_lists_tools():
    server = build_server([TOOL])
    assert server.name == "aao-ferramentas"


def test_tools_file_roundtrip(tmp_path):
    from app.mcp_tools_server import load_tools

    f = tmp_path / "tools.json"
    f.write_text(json.dumps([TOOL]))
    assert load_tools(str(f))[0]["name"] == "cadastrar_usuario"
