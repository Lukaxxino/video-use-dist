"""Auto-starts and stops the local Canary ASR server for `main.py`'s plain
CLI path, reusing `ServiceSupervisor`'s already-proven probe/start/stop
machinery -- the same machinery the panel's model catalog already uses for
whisper/nemo/canary/ollama -- without requiring an editor workstation to
adopt the panel's full `AI_AUDIO_MODELS`/`MODEL_<ID>_*` catalog schema in
its `.env`. This builds the one `ModelSpec` it needs directly from the
existing, simpler `NEMO_URL` contract `helpers/transcribe.py` already reads.

Scope is deliberately narrow:
- Only ever starts `helpers/canary_server.py`.
- Only when `NEMO_URL` resolves to localhost/127.0.0.1. A remote/on-prem
  `NEMO_URL` (e.g. a GH200 endpoint) is never touched; if it is
  unreachable, the real `helpers/transcribe.py` call still fails with its
  own specific connection error, exactly as before this module existed.
- Only when nothing is already listening and healthy there. An
  already-running server -- started by hand, by another tool, or left warm
  from a previous run -- is reused and left running; a lease that did not
  start a service never stops it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlsplit

from helpers.env_config import get_env
from helpers.model_catalog import ModelRole, ModelSpec
from helpers.service_supervisor import (
    HttpServiceProbe,
    OwnedProcessLauncher,
    ServiceSupervisor,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1"})
_DEFAULT_CANARY_PORT = 8002


def _canary_spec(nemo_url: str) -> Optional[ModelSpec]:
    """Builds the `ModelSpec` for the Canary server `nemo_url` should
    reach, or `None` if `nemo_url` is not a local address this process
    could start anything for."""
    base_url = nemo_url.rstrip("/")
    if base_url.endswith("/v1/transcribe"):
        base_url = base_url[: -len("/v1/transcribe")]
    parsed = urlsplit(base_url)
    if parsed.hostname not in _LOCAL_HOSTS:
        return None
    port = parsed.port or _DEFAULT_CANARY_PORT
    server_script = _REPO_ROOT / "helpers" / "canary_server.py"
    command = f'"{sys.executable}" "{server_script}" --port {port}'
    return ModelSpec(
        id="canary_autostart",
        label="Canary ASR (auto-started)",
        role=ModelRole.AUDIO,
        backend="canary",
        model="nvidia/canary-1b-v2",
        url=base_url,
        command=command,
    )


@contextlib.contextmanager
def ensure_canary_ready(
    asr_backend: str,
    *,
    supervisor: Optional[ServiceSupervisor] = None,
) -> Iterator[None]:
    """No-op for any `asr_backend` other than `"nemo"` (this project's
    contract name for the Canary/Parakeet-shaped JSON API -- see
    `helpers/transcribe.py`'s `ASR_BACKENDS`), for an unset `NEMO_URL`, or
    for a `NEMO_URL` that is not local (see module docstring). Starting a
    service this call did not itself launch is never attempted -- an
    already-healthy endpoint is reused and left exactly as found.

    `supervisor` is for tests; production callers never pass it.
    """
    if asr_backend != "nemo":
        yield
        return

    nemo_url = get_env("NEMO_URL")
    if not nemo_url:
        yield
        return

    spec = _canary_spec(nemo_url)
    if spec is None:
        yield
        return

    owns_supervisor = supervisor is None
    if owns_supervisor:
        probe = HttpServiceProbe()
        supervisor = ServiceSupervisor(probe, OwnedProcessLauncher(probe))

    try:
        lease = asyncio.run(supervisor.ensure_ready(spec))
    except Exception:
        # A failed ensure_ready here (server script missing, port
        # unexpectedly occupied by something else, etc.) must not block a
        # caller whose real need is just the eventual transcribe.py call --
        # that call fails with its own specific, already-sharp error if the
        # endpoint truly never comes up. This module only ever tries to
        # help start it; it never gets to be the reason a run stops early.
        logger.warning(
            "could not confirm/start the local Canary server at %s; "
            "continuing, the ASR call itself will report the real error "
            "if it truly is unreachable", spec.url, exc_info=True
        )
        yield
        return

    if lease.reused:
        print(f"[ASR AUTO] Reusing already-running Canary server at {spec.url}.")
    else:
        print(f"[ASR AUTO] Started local Canary server at {spec.url} (pid={lease.pid}).")

    try:
        yield
    finally:
        if not lease.reused:
            print(f"[ASR AUTO] Stopping the Canary server this run started at {spec.url}.")
            try:
                asyncio.run(supervisor.close_owned())
            except Exception:
                logger.warning(
                    "failed to stop the auto-started Canary server at %s; "
                    "it may still be running and holding GPU memory",
                    spec.url, exc_info=True,
                )
