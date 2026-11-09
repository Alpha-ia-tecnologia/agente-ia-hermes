# aao-hermes-bridge — FastAPI + instâncias Hermes por tenant
# O hermes-agent (nosso fork, com os patches do AAO) é clonado no build.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates bubblewrap \
  && rm -rf /var/lib/apt/lists/*

# uv gerencia os dois projetos (bridge e hermes)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# fork do hermes-agent (contém os patches do AAO — não usar o upstream puro)
ARG HERMES_REPO_URL=https://github.com/alphatechma/atch-aao-hermes-agent.git
ARG HERMES_REF=main
# repo privado: passe HERMES_GIT_TOKEN (PAT com read-only de conteúdo)
ARG HERMES_GIT_TOKEN=""
RUN CLONE_URL="${HERMES_REPO_URL}"; \
    if [ -n "${HERMES_GIT_TOKEN}" ]; then \
      CLONE_URL=$(echo "${HERMES_REPO_URL}" | sed "s#https://#https://x-access-token:${HERMES_GIT_TOKEN}@#"); \
    fi; \
    git clone --depth 1 --branch "${HERMES_REF}" "${CLONE_URL}" /opt/hermes-agent
# extras necessários: acp (adaptador que o bridge spawna — sem ele o processo
# morre com "No module named 'acp'") e mcp (as ferramentas da empresa chegam
# via servidor MCP em toda sessão)
RUN cd /opt/hermes-agent && \
    (uv sync --frozen --extra acp --extra mcp || uv sync --extra acp --extra mcp)

WORKDIR /opt/aao-hermes-bridge
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen || uv sync
COPY . .

ENV AAO_BRIDGE_HERMES_REPO_PATH=/opt/hermes-agent \
    AAO_BRIDGE_TENANTS_DIR=/shared/bridge-data/tenants \
    PYTHONUNBUFFERED=1

EXPOSE 8100
# keep-alive maior que o do cliente (httpx/undici, ~5 s): com o padrão do
# uvicorn (5 s) o servidor fechava a conexão ociosa no instante em que o cliente
# a reaproveitava -> ReadError e chamada MCP perdida (~1 em 1.400 sob carga)
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100", "--timeout-keep-alive", "65"]
