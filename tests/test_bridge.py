from app.hermes_manager import TenantInstance
from app.main import _translate_update
from app.models import ProviderConfig, SandboxConfig


def test_translate_message_chunk():
    ev = _translate_update(
        {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "olá"}},
        agent_id="a1",
    )
    assert ev is not None and ev.type == "token_chunk" and ev.text == "olá" and ev.agent_id == "a1"


def test_translate_tool_call_and_delegation():
    tool = _translate_update(
        {"sessionUpdate": "tool_call", "toolCallId": "t1", "title": "Lendo PDF", "kind": "read"},
        agent_id="a1",
    )
    assert tool is not None and tool.type == "tool_call" and tool.text == "Lendo PDF"

    delegation = _translate_update(
        {"sessionUpdate": "tool_call", "toolCallId": "t2", "title": "delegate_task para Financeiro"},
        agent_id="a1",
    )
    assert delegation is not None and delegation.type == "agent_delegation"


def test_translate_unknown_returns_none():
    assert _translate_update({"sessionUpdate": "available_commands_update"}, agent_id="a1") is None


def test_provision_writes_tenant_config(tmp_path, monkeypatch):
    from app import config

    monkeypatch.setattr(config.settings, "tenants_dir", tmp_path)
    inst = TenantInstance("empresa-1")
    inst.home = tmp_path / "empresa-1"
    inst.provision(
        ProviderConfig(provider="deepseek", model="deepseek-chat", api_key="sk-teste"),
        sandbox=SandboxConfig(),
    )

    cfg = (inst.home / "config.yaml").read_text()
    assert 'default: "deepseek-chat"' in cfg
    assert 'provider: "deepseek"' in cfg
    assert 'api_key: "sk-teste"' in cfg
    assert "system_prompt:" in cfg  # guardrail da empresa continua no config
    # sandbox padrão: sem terminal
    assert "terminal" not in cfg.split("model:")[0]
    assert "nunca exclua" in cfg

    env = (inst.home / ".env").read_text()
    assert "DEEPSEEK_API_KEY=sk-teste" in env
    assert (inst.home / "workspace").is_dir()
    assert inst.session_cwd == str(inst.home / "workspace")


def test_provision_custom_sandbox(tmp_path, monkeypatch):
    from app import config

    monkeypatch.setattr(config.settings, "tenants_dir", tmp_path)
    inst = TenantInstance("empresa-2")
    inst.home = tmp_path / "empresa-2"
    inst.provision(
        ProviderConfig(provider="deepseek", model="deepseek-chat", api_key="sk-teste"),
        sandbox=SandboxConfig(mode="custom", path="/home/cliente/projetos"),
    )
    cfg = (inst.home / "config.yaml").read_text()
    assert '"terminal"' in cfg
    assert "/home/cliente/projetos" in cfg
    assert inst.session_cwd == "/home/cliente/projetos"


def test_agent_prompt_carries_identity():
    from app.hermes_manager import agent_prompt

    text = agent_prompt("Financeiro", "Seja objetivo.")
    assert "Seu nome é Financeiro" in text and text.endswith("Seja objetivo.")
    assert agent_prompt("", "") == ""


class _FakeClient:
    def __init__(self):
        self.process = type("P", (), {"returncode": None})()
        self.closed = False

    async def close(self):
        self.closed = True


def test_admission_caps_and_lru_eviction(monkeypatch):
    import asyncio

    from app import config
    from app.hermes_manager import HermesManager

    monkeypatch.setattr(config.settings, "max_instances", 2)
    monkeypatch.setattr(config.settings, "max_turns_per_instance", 2)
    monkeypatch.setattr(config.settings, "max_active_turns", 3)

    async def run():
        m = HermesManager()
        a, b, c = m.get("a"), m.get("b"), m.get("c")
        a.client, b.client = _FakeClient(), _FakeClient()
        a.last_used, b.last_used = 1.0, 2.0

        # por instância: 2 turnos, o 3º espera
        assert m.try_admit(a) and m.try_admit(a)
        assert not m.try_admit(a)
        # b ocupa a 2ª vaga; global = 3
        assert m.try_admit(b)
        # global cheio
        assert not m.try_admit(c)
        m.release(b)
        # c precisa de vaga: a está em turno, b ocioso -> b é reciclado
        old_b = b.client
        assert m.try_admit(c)
        assert b.client is None and not b.running
        await asyncio.sleep(0)
        assert old_b.closed
        # todas as vagas em turno: d espera em vez de derrubar alguém
        m.release(a)  # a ainda tem 1 turno
        d = m.get("d")
        c.client = _FakeClient()
        assert not m.try_admit(d)

    # loop próprio: asyncio.run() zeraria o loop global usado por outros testes
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(run())
    finally:
        loop.close()


def test_queue_is_fifo_without_head_of_line_blocking(monkeypatch):
    import asyncio

    from app import config
    from app.hermes_manager import HermesManager

    monkeypatch.setattr(config.settings, "max_instances", 5)
    monkeypatch.setattr(config.settings, "max_turns_per_instance", 1)
    monkeypatch.setattr(config.settings, "max_active_turns", 2)

    async def run():
        m = HermesManager()
        a, b, c = m.get("a"), m.get("b"), m.get("c")
        # a e b ocupam as 2 vagas globais
        assert m.enqueue(a).done() and m.enqueue(b).done()
        # fila: a2 (a já no teto da instância), c1, c2 — nesta ordem
        a2, c1, c2 = m.enqueue(a), m.enqueue(c), m.enqueue(c)
        assert [m.position(f) for f in (a2, c1, c2)] == [1, 2, 3]
        # libera uma vaga de b: a2 não cabe (a no teto) e não segura c1
        m.release(b)
        assert not a2.done() and c1.done() and not c2.done()
        # libera a: a2 chegou antes de c2 e é admitido primeiro
        m.release(a)
        assert a2.done() and not c2.done()
        # desistência na fila não vaza vaga; desistência já admitida devolve
        m.cancel(c, c2)
        assert m.position(c2) == 0
        before = m.active_turns
        m.cancel(a, a2)
        assert m.active_turns == before - 1

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(run())
    finally:
        loop.close()
