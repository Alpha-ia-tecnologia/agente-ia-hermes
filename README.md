# aao-hermes-bridge

Ponte de orquestração do **Alpha Agents Office**: FastAPI que gerencia uma
instância do Hermes (fork `aao-hermes-agent`) por empresa (tenant), com
HERMES_HOME isolado, sandbox bwrap opcional, ferramentas dinâmicas via MCP,
login por assinatura (device-code/PKCE) e endpoints de skills/aprendizados.

## Rodando

```bash
uv sync
cp .env.example .env   # AAO_BRIDGE_BRIDGE_API_KEY etc.
uv run uvicorn app.main:app --port 8100
```

Requer o repo `aao-hermes-agent` ao lado (ou `AAO_BRIDGE_HERMES_REPO_PATH`).
Compartilha o diretório `data/` com a aao-api (uploads/workspaces dos tenants).

## Capacidade e concorrência

Um processo Hermes por empresa atende vários usuários ao mesmo tempo:

- **Fila por sessão:** cada `conversa:agente` tem a sua fila de eventos e o seu
  turno; chats simultâneos da mesma empresa não se misturam.
- **Agente por sessão:** nome e prompt do agente vão por turno
  (`_meta.aaoSystemPrompt`, patch no fork) — trocar de agente não reinicia o processo.
- **Ferramentas por sessão:** servidas pelo próprio bridge em `/mcp/` (MCP
  streamable HTTP), com um token por sessão.
- **Controle de entrada:** acima dos limites o chat espera na fila, com aviso
  ao usuário a cada 10 s, em vez de derrubar o servidor.

| Variável | Padrão | O que limita |
|---|---|---|
| `AAO_BRIDGE_MAX_INSTANCES` | 6 | processos Hermes vivos (~200 MB cada em turno); a instância ociosa mais antiga é reciclada |
| `AAO_BRIDGE_MAX_TURNS_PER_INSTANCE` | 30 | turnos simultâneos num processo (uma empresa) |
| `AAO_BRIDGE_MAX_ACTIVE_TURNS` | 40 | turnos simultâneos no bridge todo |
| `AAO_BRIDGE_MAX_LIVE_SESSIONS` | 40 | sessões em memória por processo (~6 MB cada); as demais voltam do banco do Hermes |
| `AAO_BRIDGE_IDLE_TIMEOUT_SECONDS` | 300 | tempo até reciclar uma instância ociosa |
| `AAO_BRIDGE_QUEUE_TIMEOUT_SECONDS` | 300 | espera máxima na fila |
| `AAO_BRIDGE_MCP_TOOL_TIMEOUT_SECONDS` | 120 | teto de uma chamada de ferramenta |

Requer o fork `aao-hermes-agent` com os patches de ACP (`HERMES_ACP_MAX_WORKERS`,
`HERMES_ACP_MAX_LIVE_SESSIONS`, `HERMES_ACP_MCP_TOOL_TIMEOUT`, `aaoSystemPrompt`).
Faça o deploy do fork **antes** do bridge.
