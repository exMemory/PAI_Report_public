"""Project environment loader for the PAI render scripts.

Loads plain KEY=VALUE lines from a local .env file at the project root,
if present, without overriding existing environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path


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


def load_project_dotenv() -> None:
    _load_dotenv_file()
