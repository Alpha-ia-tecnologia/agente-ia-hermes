from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    hermes_repo_path: Path = Path(__file__).resolve().parents[2] / "hermes-agent"
    tenants_dir: Path = Path(__file__).resolve().parents[1] / "data" / "tenants"
    # Instância ociosa é reciclada depois disso (libera ~170 MB por empresa).
    # 30 min: reaquecer custa ~10 s na 1ª mensagem; max_instances limita a RAM.
    idle_timeout_seconds: int = 1800
    session_new_timeout_seconds: int = 60
    prompt_timeout_seconds: int = 600
    # Quanto esperar turnos ativos terminarem antes de reiniciar a instância
    # por mudança de config (troca de provider/sandbox).
    restart_drain_timeout_seconds: int = 120
    bridge_api_key: str = "dev-bridge-key"
    # Sandbox duro (bubblewrap): None = auto (ativa se `bwrap` existir).
    # Desative com AAO_BRIDGE_HARD_SANDBOX=0.
    hard_sandbox: bool | None = None

    # ---- Capacidade (dimensionar pela RAM livre da VPS) ----
    # Processos Hermes vivos ao mesmo tempo (~170 MB ociosos cada, mais em turno).
    # Acima disso, a instância ociosa mais antiga é reciclada; se todas estiverem
    # em turno, o chat espera na fila.
    max_instances: int = 6
    # Turnos simultâneos dentro de UM processo (threads do Hermes — vira
    # HERMES_ACP_MAX_WORKERS). Limita quanto uma empresa sozinha consome.
    max_turns_per_instance: int = 30
    # Sessões (conversa:agente) mantidas em memória por processo — cada uma
    # custa ~6 MB; as demais ficam no banco do Hermes e voltam no próximo turno.
    max_live_sessions: int = 40
    # Turnos simultâneos no bridge inteiro (todas as empresas).
    max_active_turns: int = 40
    # Teto de uma chamada de ferramenta da empresa (MCP) dentro do Hermes.
    mcp_tool_timeout_seconds: int = 120
    # Quanto um chat pode esperar na fila antes de desistir.
    queue_timeout_seconds: int = 300
    # URL pela qual os processos Hermes (filhos deste bridge) alcançam o
    # servidor MCP de ferramentas servido pelo próprio bridge.
    self_url: str = "http://127.0.0.1:8100"

    model_config = {"env_prefix": "AAO_BRIDGE_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
