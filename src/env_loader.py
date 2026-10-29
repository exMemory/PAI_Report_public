"""Project environment loader for the PAI render scripts.

Two sources, in order:
1. A local .env file (project root), if present -- plain KEY=VALUE lines.
2. Muse's Secure Vault: for providers registered as custom connectors,
   fetch the `hsurr:` surrogate from authd and expose it under the env var
   the render script already reads. The real credential is swapped in by
   the runtime on outbound requests to the registered host; the surrogate
   itself is not a secret and is never printed or logged here.

Each provider is loaded independently and best-effort: if a provider is
not connected, its env var is simply left unset and the caller's existing
"requires ..." check reports it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROVIDERS = {
    "OPENROUTER_API_KEY": "custom.openrouter",
    "GEMINI_API_KEY": "custom.google-gemini",
}


def _load_dotenv_file() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def _load_vault_surrogates() -> None:
    sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
    try:
        from dynamic_credentials import dynamic_credential_entry
    except Exception:
        return
    for env_name, credential_name in _PROVIDERS.items():
        if os.environ.get(env_name):
            continue
        try:
            entry = dynamic_credential_entry(credential_name, "access_token")
        except Exception:
            continue
        surrogate = str(entry.get("surrogate", "")).strip()
        if surrogate.startswith("hsurr:"):
            os.environ[env_name] = surrogate


def load_project_dotenv() -> None:
    _load_dotenv_file()
    _load_vault_surrogates()
