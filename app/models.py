from typing import Literal, Optional

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Provider/model resolved by aao-api for this request (BYOK)."""

    provider: Literal[
        "openai",
        "anthropic",
        "deepseek",
        "google",
        "openrouter",
        "custom",
        # assinaturas: credenciais vivem no auth store do tenant (HERMES_HOME),
        # não em api_key — ver hermes_auth_helper/subscription_auth.
        "openai-codex",
        "anthropic-oauth",
        "copilot",
        "nous",
        "xai-oauth",
    ]
    model: str
    api_key: str = ""
    base_url: Optional[str] = None
    temperature: Optional[float] = None


class ToolDef(BaseModel):
    """Ferramenta dinâmica cadastrada pelo cliente (function calling via MCP)."""

    name: str = Field(pattern=r"^[a-z0-9_]{2,64}$")
    description: str
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "POST"
    url: str
    headers: dict[str, str] = {}
    params_schema: dict = {"type": "object", "properties": {}}
    timeout_ms: int = 15000


class SandboxConfig(BaseModel):
    """Escopo de atuação dos agentes, definido pela empresa.

    - workspace: só o workspace do tenant (padrão, sem terminal)
    - custom: pasta escolhida pelo usuário (com terminal, sem exclusões)
    - free: host inteiro (com terminal, sem exclusões)
    """

    mode: Literal["workspace", "custom", "free"] = "workspace"
    path: Optional[str] = None


class AgentConfig(BaseModel):
    agent_id: str
    name: str
    system_prompt: str = ""


class ChatRequest(BaseModel):
    tenant_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    session_id: Optional[str] = None
    message: str
    file_paths: list[str] = []
    agent: AgentConfig
    # provedor efetivo deste turno (o do agente, ou a chave reserva)
    provider: ProviderConfig
    # provedor padrão da empresa: é o que vai no config do processo Hermes.
    # Quando `provider` difere, ele vale só para esta sessão (sem reiniciar).
    tenant_provider: Optional[ProviderConfig] = None
    tools: list[ToolDef] = []
    sandbox: SandboxConfig = SandboxConfig()


class BridgeEvent(BaseModel):
    """Normalized event re-emitted to aao-api as NDJSON."""

    type: Literal[
        "session_started",
        "agent_status",
        "agent_delegation",
        "tool_call",
        "tool_call_update",
        "token_chunk",
        "thought_chunk",
        "usage",
        "final_message",
        "error",
    ]
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    text: Optional[str] = None
    data: Optional[dict] = None
