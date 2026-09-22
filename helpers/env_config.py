"""Central place to read configuration from the environment or the repo-root `.env`.

Every backend (ASR server, Ollama vision, benchmark CSV location, deployment label) is configured
here so the pipeline can point at a local dev server today and an on-prem/remote server tomorrow
by only editing `.env`. Real environment variables take precedence over `.env`.
"""

from __future__ import annotations

import os
from pathlib import Path

# helpers/ -> repo root
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def get_env(key: str, default: str = "") -> str:
    """Return an env var, falling back to the repo-root `.env`, then `default`."""
    val = os.environ.get(key)
    if val:
        return val
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{key}=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return default
