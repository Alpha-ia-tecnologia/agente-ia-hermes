"""Helper de autenticação por assinatura, executado com HERMES_HOME do tenant.

Roda como subprocess (`uv run --project hermes-agent --directory hermes-agent
python <este arquivo> <cmd>`) porque os módulos do Hermes resolvem HERMES_HOME
no import — cada invocação atende exatamente um tenant.

Protocolo: eventos NDJSON no stdout:
  {"event": "verification", "uri": "...", "code": "..."}   (fluxos device-code)
  {"event": "done", "provider": "..."}
  {"event": "error", "message": "..."}
Comandos síncronos (status/url/exchange/import/remove) imprimem um JSON único.
"""

from __future__ import annotations

import builtins
import io
import json
import re
import sys
import uuid


def emit(payload: dict) -> None:
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
URL_RE = re.compile(r"https?://\S+")
# código no meio da linha ("Enter this code: 068E-BE43") — device codes têm hífen
CODE_INLINE_RE = re.compile(r"\b[A-Z0-9]{4,8}-[A-Z0-9]{4,8}\b")
# fallback: a linha inteira é o código (formatos sem hífen)
CODE_RE = re.compile(r"^[A-Z0-9]{4,10}(-[A-Z0-9]{4,10})*$")


class PrintInterceptor:
    """Captura URL de verificação + código dos prints dos fluxos do Hermes."""

    # sinais de que a URL é mesmo a de verificação (e não um link de intro)
    STRONG_URL_HINTS = ("device", "user_code", "verify", "activate", "code=")

    def __init__(self) -> None:
        self.uri: str | None = None
        self.code: str | None = None
        self._strong_uri = False
        self._emitted = False
        self._original_print = builtins.print

    def __enter__(self):
        def _hook(*args, **kwargs):
            buf = io.StringIO()
            kwargs.pop("file", None)
            self._original_print(*args, file=buf, **kwargs)
            text = ANSI_RE.sub("", buf.getvalue())
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if not self._strong_uri:
                    m = URL_RE.search(line)
                    if m:
                        url = m.group(0).rstrip(".,")
                        strong = any(h in url.lower() for h in self.STRONG_URL_HINTS)
                        # pega a 1ª URL, mas troca por uma "forte" se aparecer
                        if self.uri is None or strong:
                            self.uri = url
                            self._strong_uri = strong
                if self.code is None:
                    cm = CODE_INLINE_RE.search(line)
                    if cm:
                        self.code = cm.group(0)
                    elif CODE_RE.match(line):
                        self.code = line
            if self.uri and self.code and not self._emitted:
                self._emitted = True
                emit({"event": "verification", "uri": self.uri, "code": self.code})
            # logs originais vão para o stderr (visíveis no log do bridge)
            sys.stderr.write(text)

        builtins.print = _hook
        return self

    def __exit__(self, *exc):
        builtins.print = self._original_print
        return False

    def emit_now(self, uri: str, code: str) -> None:
        if not self._emitted:
            self._emitted = True
            emit({"event": "verification", "uri": uri, "code": code})


# ---- Fluxos device-code (bloqueantes até o usuário autorizar) --------------


def flow_openai_codex() -> None:
    from hermes_cli.auth import _codex_device_code_login, _save_codex_tokens

    with PrintInterceptor():
        creds = _codex_device_code_login()
    _save_codex_tokens(creds["tokens"], creds.get("last_refresh"))
    emit({"event": "done", "provider": "openai-codex"})


def flow_nous() -> None:
    from hermes_cli.auth import (
        _auth_store_lock,
        _load_auth_store,
        _nous_device_code_login,
        _save_auth_store,
        _save_provider_state,
    )

    with PrintInterceptor() as icp:
        state = _nous_device_code_login(
            open_browser=False,
            on_verification=lambda uri, code: icp.emit_now(uri, code),
        )
    with _auth_store_lock():
        store = _load_auth_store()
        _save_provider_state(store, "nous", state)
        _save_auth_store(store)
    try:
        from agent.credential_sources import unsuppress_credential_source

        unsuppress_credential_source("nous", "device_code")
    except Exception:
        pass
    emit({"event": "done", "provider": "nous"})


def flow_xai() -> None:
    from hermes_cli.auth import _save_xai_oauth_tokens, _xai_oauth_device_code_login

    with PrintInterceptor():
        creds = _xai_oauth_device_code_login(open_browser=False)
    _save_xai_oauth_tokens(
        creds["tokens"],
        discovery=creds.get("discovery"),
        redirect_uri=creds.get("redirect_uri", ""),
        last_refresh=creds.get("last_refresh"),
        auth_mode="oauth_device_code",
    )
    try:
        from agent.credential_sources import unsuppress_credential_source

        unsuppress_credential_source("xai-oauth", "device_code")
    except Exception:
        pass
    emit({"event": "done", "provider": "xai-oauth"})


def flow_copilot() -> None:
    from hermes_cli.copilot_auth import copilot_device_code_login
    from hermes_cli.config import save_env_value

    with PrintInterceptor():
        token = copilot_device_code_login(timeout_seconds=900)
    if not token:
        emit({"event": "error", "message": "Login do Copilot cancelado ou expirado."})
        sys.exit(1)
    save_env_value("COPILOT_GITHUB_TOKEN", token)
    emit({"event": "done", "provider": "copilot"})


FLOWS = {
    "openai-codex": flow_openai_codex,
    "nous": flow_nous,
    "xai-oauth": flow_xai,
    "copilot": flow_copilot,
}


# ---- Anthropic (Claude Pro/Max): PKCE em duas etapas ------------------------


def _pkce_state_file():
    from hermes_constants import get_hermes_home

    return get_hermes_home() / ".aao_anthropic_pkce.json"


def anthropic_url() -> None:
    import secrets
    from urllib.parse import urlencode

    from agent.anthropic_adapter import (
        _generate_pkce,
        _OAUTH_CLIENT_ID,
        _OAUTH_REDIRECT_URI,
        _OAUTH_SCOPES,
    )

    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(32)
    params = {
        "code": "true",
        "client_id": _OAUTH_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _OAUTH_REDIRECT_URI,
        "scope": _OAUTH_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    f = _pkce_state_file()
    f.write_text(json.dumps({"verifier": verifier, "state": state}))
    f.chmod(0o600)
    emit({"auth_url": f"https://claude.ai/oauth/authorize?{urlencode(params)}"})


def _persist_anthropic_tokens(access_token: str, refresh_token: str, expires_at_ms: int) -> None:
    from agent.credential_pool import (
        AUTH_TYPE_OAUTH,
        SOURCE_MANUAL,
        PooledCredential,
        load_pool,
    )

    pool = load_pool("anthropic")
    # remove entradas anteriores desta origem para não acumular duplicatas
    for i, entry in reversed(list(enumerate(pool.entries(), start=1))):
        if entry.source == f"{SOURCE_MANUAL}:hermes_pkce":
            try:
                pool.remove_index(i)
            except Exception:
                pass
    pool.add_entry(
        PooledCredential(
            provider="anthropic",
            id=uuid.uuid4().hex[:6],
            label="Assinatura Claude (AAO)",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=f"{SOURCE_MANUAL}:hermes_pkce",
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at_ms=expires_at_ms,
        )
    )


def anthropic_exchange(pasted_code: str) -> None:
    import time
    import urllib.request

    from agent.anthropic_adapter import (
        _OAUTH_CLIENT_ID,
        _OAUTH_REDIRECT_URI,
        _OAUTH_TOKEN_URLS,
        _OAUTH_TOKEN_USER_AGENT,
    )

    f = _pkce_state_file()
    if not f.exists():
        emit({"event": "error", "message": "Fluxo não iniciado — gere a URL primeiro."})
        sys.exit(1)
    pkce = json.loads(f.read_text())

    splits = pasted_code.strip().split("#")
    code = splits[0]
    received_state = splits[1] if len(splits) > 1 else ""
    if received_state and received_state != pkce["state"]:
        emit({"event": "error", "message": "Código inválido (state não confere). Gere uma nova URL."})
        sys.exit(1)

    body = json.dumps(
        {
            "grant_type": "authorization_code",
            "client_id": _OAUTH_CLIENT_ID,
            "code": code,
            "state": received_state,
            "redirect_uri": _OAUTH_REDIRECT_URI,
            "code_verifier": pkce["verifier"],
        }
    ).encode()
    result = None
    last_error: Exception | None = None
    for endpoint in _OAUTH_TOKEN_URLS:
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": _OAUTH_TOKEN_USER_AGENT},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode())
            break
        except Exception as exc:  # tenta o próximo endpoint
            last_error = exc
    if result is None or not result.get("access_token"):
        emit({"event": "error", "message": f"Troca do código falhou: {last_error}"})
        sys.exit(1)

    expires_at_ms = int(time.time() * 1000) + int(result.get("expires_in", 3600)) * 1000
    _persist_anthropic_tokens(
        result["access_token"], result.get("refresh_token", ""), expires_at_ms
    )
    f.unlink(missing_ok=True)
    emit({"event": "done", "provider": "anthropic"})


# ---- Import (credenciais vindas do conector local) --------------------------


def do_import(source: str) -> None:
    payload = json.loads(sys.stdin.read() or "{}")
    if source == "claude-code":
        oauth = payload.get("claudeAiOauth") or payload
        access = oauth.get("accessToken") or oauth.get("access_token")
        refresh = oauth.get("refreshToken") or oauth.get("refresh_token") or ""
        expires = int(oauth.get("expiresAt") or oauth.get("expires_at_ms") or 0)
        if not access:
            emit({"event": "error", "message": "Credencial do Claude Code sem accessToken."})
            sys.exit(1)
        _persist_anthropic_tokens(access, refresh, expires)
        emit({"event": "done", "provider": "anthropic"})
    elif source == "codex-cli":
        from hermes_cli.auth import _save_codex_tokens

        tokens = payload.get("tokens") or payload
        if not tokens.get("access_token"):
            emit({"event": "error", "message": "Credencial do Codex CLI sem access_token."})
            sys.exit(1)
        _save_codex_tokens(
            {
                "access_token": tokens["access_token"],
                "refresh_token": tokens.get("refresh_token", ""),
            },
            payload.get("last_refresh"),
        )
        emit({"event": "done", "provider": "openai-codex"})
    elif source == "gh-token":
        from hermes_cli.config import save_env_value

        token = payload.get("token", "")
        if not token:
            emit({"event": "error", "message": "Token do GitHub vazio."})
            sys.exit(1)
        save_env_value("COPILOT_GITHUB_TOKEN", token)
        emit({"event": "done", "provider": "copilot"})
    else:
        emit({"event": "error", "message": f"Origem de import desconhecida: {source}"})
        sys.exit(1)


# ---- Status / remoção --------------------------------------------------------


def status() -> None:
    import time

    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    out: dict[str, dict] = {}

    # auth.json (codex, nous, xai)
    providers = {}
    auth_file = home / "auth.json"
    if auth_file.exists():
        try:
            providers = json.loads(auth_file.read_text()).get("providers", {})
        except Exception:
            providers = {}
    for key in ("openai-codex", "nous", "xai-oauth"):
        state = providers.get(key) or {}
        tokens = state.get("tokens") or {}
        out[key] = {"connected": bool(tokens.get("access_token") or tokens.get("refresh_token"))}

    # anthropic: pool
    try:
        from agent.credential_pool import load_pool

        pool = load_pool("anthropic")
        entries = [e for e in pool.entries() if "hermes_pkce" in (e.source or "")]
        info: dict = {"connected": bool(entries)}
        if entries and entries[0].expires_at_ms:
            info["expires_in_s"] = max(0, int(entries[0].expires_at_ms / 1000 - time.time()))
        out["anthropic"] = info
    except Exception:
        out["anthropic"] = {"connected": False}

    # copilot: env
    env_file = home / ".env"
    has_copilot = env_file.exists() and "COPILOT_GITHUB_TOKEN=" in env_file.read_text()
    out["copilot"] = {"connected": has_copilot}

    emit({"providers": out})


def remove(provider: str) -> None:
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    if provider in ("openai-codex", "nous", "xai-oauth"):
        from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store

        with _auth_store_lock():
            store = _load_auth_store()
            (store.get("providers") or {}).pop(provider, None)
            _save_auth_store(store)
    elif provider == "anthropic":
        from agent.credential_pool import load_pool

        pool = load_pool("anthropic")
        for i, entry in reversed(list(enumerate(pool.entries(), start=1))):
            if "hermes_pkce" in (entry.source or ""):
                try:
                    pool.remove_index(i)
                except Exception:
                    pass
        (home / ".anthropic_oauth.json").unlink(missing_ok=True)
    elif provider == "copilot":
        env_file = home / ".env"
        if env_file.exists():
            lines = [
                l for l in env_file.read_text().splitlines()
                if not l.startswith("COPILOT_GITHUB_TOKEN=")
            ]
            env_file.write_text("\n".join(lines) + ("\n" if lines else ""))
    else:
        emit({"event": "error", "message": f"Provider desconhecido: {provider}"})
        sys.exit(1)
    emit({"event": "done", "provider": provider})


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        if cmd == "flow":
            provider = sys.argv[2]
            if provider not in FLOWS:
                emit({"event": "error", "message": f"Fluxo não suportado: {provider}"})
                sys.exit(1)
            FLOWS[provider]()
        elif cmd == "anthropic-url":
            anthropic_url()
        elif cmd == "anthropic-exchange":
            anthropic_exchange(sys.argv[2])
        elif cmd == "import":
            do_import(sys.argv[2])
        elif cmd == "status":
            status()
        elif cmd == "remove":
            remove(sys.argv[2])
        else:
            emit({"event": "error", "message": f"Comando desconhecido: {cmd}"})
            sys.exit(2)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — superfície única de erro p/ o bridge
        emit({"event": "error", "message": str(exc)[:500]})
        sys.exit(1)


if __name__ == "__main__":
    main()
