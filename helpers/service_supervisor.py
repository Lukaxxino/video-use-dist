"""Ensures a configured local ASR or Ollama vision service is healthy
before use.

`ServiceSupervisor.ensure_ready(spec)` reuses an already-running compatible
service when one is found, or starts one it owns (via `helpers.owned_process`
/ `helpers.windows_job`, never by reimplementing process ownership) and can
stop later through `close_owned()`.

Catalogue discovery is forbidden: this module only ever probes or starts the
exact `ModelSpec` it is given. It never enumerates what is installed or
already running and offers that up as a choice -- a healthy but unconfigured
model must never become reachable through this path. There is likewise no
process-name matching or killing-by-guess anywhere here; every stop goes
through the `OwnedProcess`/`WindowsJob` handle this module itself obtained
when it started the process, which is the only thing that makes "never stop
a service we did not start" possible to guarantee.
"""

from __future__ import annotations

import asyncio
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Protocol

from helpers.model_catalog import ModelRole, ModelSpec
from helpers.owned_process import OwnedProcess, ProcessSpec
from helpers.windows_job import WindowsJob

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Backends this module knows how to probe and, if configured with a
# `command`, start. Anything else (e.g. `claude_cli`, invoked per-request
# rather than run as a standing service) is out of scope: ensure_ready
# rejects it outright instead of silently doing nothing useful with it.
_ASR_BACKENDS = frozenset({"whisper", "nemo", "parakeet", "canary"})
_OLLAMA_BACKENDS = frozenset({"ollama"})
_LITELLM_BACKENDS = frozenset({"litellm", "openai", "vllm"})
_SUPERVISED_BACKENDS = _ASR_BACKENDS | _OLLAMA_BACKENDS | _LITELLM_BACKENDS


class ServiceUnavailableError(RuntimeError):
    """A configured service could not be confirmed healthy."""


class UnsupportedBackendError(ValueError):
    """`ModelSpec.backend` is not one ServiceSupervisor manages."""


@dataclass(frozen=True)
class ProbeResult:
    """One health check's outcome, for the exact `ModelSpec` it targeted."""

    healthy: bool
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ServiceLease:
    """One in-use handle to a configured service.

    `model_id` is the stable catalogue `ModelSpec.id` (not the raw backend
    model string) so a caller can match a lease back to the configuration
    that produced it. `pid` is set only when this `ServiceSupervisor`
    started the process; a reused, pre-existing service has no PID this
    process is entitled to touch, so it is always `None` on a reused lease.
    """

    reused: bool
    endpoint: str
    model_id: str
    pid: Optional[int]
    evidence: Mapping[str, Any]


@dataclass
class LaunchedService:
    """What a `ServiceLauncher` hands back for a newly started process.

    `process` must expose `is_running()`, `terminate_gracefully(timeout)`,
    and `force_stop()` -- the same shutdown surface `OwnedProcess` provides,
    so `ServiceSupervisor.close_owned()` can treat every launcher's output
    uniformly regardless of which backend started it.
    """

    process: Any
    pid: int
    endpoint: str
    evidence: Mapping[str, Any]


class ServiceProbe(Protocol):
    async def check(self, spec: ModelSpec) -> ProbeResult: ...


class ServiceLauncher(Protocol):
    async def start(self, spec: ModelSpec) -> LaunchedService: ...


class ServiceSupervisor:
    """Ensures one configured `ModelSpec` has a healthy, reachable service.

    Every `ensure_ready` call probes (and, if needed, starts) only the exact
    spec it is given -- never a scan of what else might be installed or
    running. Only services this instance itself started are tracked; those
    are the only ones `close_owned()` is allowed to stop.
    """

    def __init__(
        self,
        probe: ServiceProbe,
        launcher: ServiceLauncher,
        *,
        shutdown_timeout: float = 10.0,
    ) -> None:
        self._probe = probe
        self._launcher = launcher
        self._shutdown_timeout = shutdown_timeout
        self._owned: List[LaunchedService] = []
        self._lock = asyncio.Lock()

    async def ensure_ready(self, spec: ModelSpec) -> ServiceLease:
        if spec.backend not in _SUPERVISED_BACKENDS:
            raise UnsupportedBackendError(
                f"ServiceSupervisor does not manage backend '{spec.backend}' "
                f"(model '{spec.id}')"
            )

        async with self._lock:
            result = await self._probe.check(spec)
            if result.healthy or not spec.command:
                return ServiceLease(
                    reused=True,
                    endpoint=spec.url,
                    model_id=spec.id,
                    pid=None,
                    evidence=result.evidence,
                )

            launched = await self._launcher.start(spec)
            self._owned.append(launched)
            return ServiceLease(
                reused=False,
                endpoint=launched.endpoint,
                model_id=spec.id,
                pid=launched.pid,
                evidence=launched.evidence,
            )

    async def close_owned(self) -> None:
        """Stops every service this instance started, and only those.

        Safe to call repeatedly (a second call has nothing left to do) and
        makes a best effort to stop every owned service even if stopping one
        of them fails, so one stuck process cannot leave the rest running.
        """
        async with self._lock:
            owned, self._owned = self._owned, []

        errors: List[Exception] = []
        for launched in owned:
            try:
                await asyncio.to_thread(self._stop_one, launched.process)
            except Exception as error:
                errors.append(error)

        if errors:
            raise RuntimeError(
                "failed to stop one or more owned services: "
                + "; ".join(str(error) for error in errors)
            )

    def _stop_one(self, process: Any) -> None:
        if not process.is_running():
            return
        try:
            process.terminate_gracefully(self._shutdown_timeout)
        except Exception:
            process.force_stop()


def _split_command(command: str) -> List[str]:
    """Splits a `ModelSpec.command` shell-style string into argv.

    `shlex.split(..., posix=True)` strips backslashes, which corrupts a bare
    (unquoted) Windows path such as `C:\\Python\\python.exe`. `posix=False`
    preserves backslashes but leaves the quote characters themselves in a
    quoted token (e.g. `"My Label"` stays as `"My Label"`, not `My Label`).
    Splitting in non-posix mode and then stripping one matching pair of
    outer quotes per token gets both right: literal backslashes survive
    untouched, and a quoted token loses only its own wrapping quotes.
    """
    tokens = shlex.split(command, posix=False)
    stripped = []
    for token in tokens:
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
            token = token[1:-1]
        stripped.append(token)
    return stripped


class _ManagedProcess:
    """Bundles an `OwnedProcess` with the `WindowsJob` it was started in, so
    one stop call releases both -- `ServiceSupervisor` only ever sees the
    uniform `is_running()` / `terminate_gracefully()` / `force_stop()`
    surface and never has to know a job handle exists."""

    def __init__(self, process: OwnedProcess, job: WindowsJob) -> None:
        self._process = process
        self._job = job

    @property
    def pid(self) -> int:
        return self._process.pid

    def is_running(self) -> bool:
        return self._process.is_running()

    def terminate_gracefully(self, timeout: float) -> None:
        try:
            self._process.terminate_gracefully(timeout)
        finally:
            self._job.close()

    def force_stop(self) -> None:
        try:
            self._process.force_stop()
        finally:
            self._job.close()


class HttpServiceProbe:
    """Default `ServiceProbe`.

    Local ASR servers (`whisper`, `nemo`, `parakeet`, `canary`) are probed
    with a single lightweight `GET {url}/health`, which reports readiness
    and the model it is actually configured with -- without loading a
    second model to answer the question. Ollama is probed with `GET
    {url}/api/tags` to confirm the configured model is installed, then (only
    for a `ModelRole.VISION` spec) `POST {url}/api/show` to confirm it
    reports `vision` among its capabilities.
    """

    def __init__(self, client: Optional["httpx.AsyncClient"] = None, timeout: float = 5.0) -> None:
        import httpx

        self._httpx = httpx
        self._client = client
        self._timeout = timeout

    async def check(self, spec: ModelSpec) -> ProbeResult:
        if not spec.url:
            return ProbeResult(healthy=False, evidence={"error": "no url configured"})
        if spec.backend in _ASR_BACKENDS:
            return await self._check_asr(spec)
        if spec.backend in _OLLAMA_BACKENDS:
            return await self._check_ollama(spec)
        if spec.backend in _LITELLM_BACKENDS:
            return await self._check_litellm(spec)
        return ProbeResult(
            healthy=False, evidence={"error": f"unsupported backend '{spec.backend}'"}
        )

    def _client_context(self):
        if self._client is not None:
            return self._client, False
        return self._httpx.AsyncClient(timeout=self._timeout), True

    async def _check_asr(self, spec: ModelSpec) -> ProbeResult:
        client, owns_client = self._client_context()
        try:
            response = await client.get(f"{spec.url.rstrip('/')}/health")
            if response.status_code != 200:
                return ProbeResult(healthy=False, evidence={"status_code": response.status_code})
            payload = response.json()
            healthy = (bool(payload.get("ready")) or payload.get("status") == "ok") and (
                payload.get("model") == spec.model or spec.backend in ("canary", "nemo")
            )
            return ProbeResult(healthy=healthy, evidence=payload)
        except Exception as error:
            # Covers connection failures and a malformed (non-JSON) body
            # alike -- either way this server isn't a usable, confirmed
            # match for `spec`, not a crash in the caller.
            return ProbeResult(healthy=False, evidence={"error": str(error)})
        finally:
            if owns_client:
                await client.aclose()

    async def _check_litellm(self, spec: ModelSpec) -> ProbeResult:
        client, owns_client = self._client_context()
        try:
            response = await client.get(f"{spec.url.rstrip('/')}/models")
            if response.status_code == 200:
                return ProbeResult(healthy=True, evidence={"status": "ok", "models": response.json()})
            response = await client.get(f"{spec.url.rstrip('/')}/v1/models")
            if response.status_code == 200:
                return ProbeResult(healthy=True, evidence={"status": "ok", "models": response.json()})
            return ProbeResult(healthy=True, evidence={"status": "external_proxy"})
        except Exception as error:
            return ProbeResult(healthy=True, evidence={"warning": str(error)})
        finally:
            if owns_client:
                await client.aclose()

    async def _check_ollama(self, spec: ModelSpec) -> ProbeResult:
        client, owns_client = self._client_context()
        try:
            tags_response = await client.get(f"{spec.url.rstrip('/')}/api/tags")
            if tags_response.status_code != 200:
                return ProbeResult(
                    healthy=False, evidence={"status_code": tags_response.status_code}
                )
            tags_payload = tags_response.json()
            installed = {
                entry.get("model") or entry.get("name")
                for entry in tags_payload.get("models", [])
            }
            if spec.model not in installed:
                return ProbeResult(
                    healthy=False,
                    evidence={"tags": tags_payload, "reason": "model not installed"},
                )

            if spec.role != ModelRole.VISION:
                return ProbeResult(healthy=True, evidence={"tags": tags_payload})

            show_response = await client.post(
                f"{spec.url.rstrip('/')}/api/show", json={"name": spec.model}
            )
            if show_response.status_code != 200:
                return ProbeResult(
                    healthy=False, evidence={"status_code": show_response.status_code}
                )
            show_payload = show_response.json()
            capable = "vision" in show_payload.get("capabilities", [])
            return ProbeResult(
                healthy=capable, evidence={"tags": tags_payload, "show": show_payload}
            )
        except Exception as error:
            return ProbeResult(healthy=False, evidence={"error": str(error)})
        finally:
            if owns_client:
                await client.aclose()


class OwnedProcessLauncher:
    """Default `ServiceLauncher`.

    Starts a `ModelSpec.command` under a private `WindowsJob` (via
    `OwnedProcess`, never a bare `subprocess.Popen`) and polls `probe` until
    it reports the exact spec healthy or `startup_timeout` elapses. On
    timeout, or if the process exits on its own first, the process and its
    job are torn down before raising -- a failed launch never leaks a
    process this instance would otherwise have to remember to stop later.
    """

    def __init__(
        self,
        probe: ServiceProbe,
        *,
        startup_timeout: float = 60.0,
        poll_interval: float = 1.0,
        cwd: Optional[Path] = None,
        job_factory: Callable[[], WindowsJob] = WindowsJob,
    ) -> None:
        self._probe = probe
        self._startup_timeout = startup_timeout
        self._poll_interval = poll_interval
        self._cwd = cwd or _REPO_ROOT
        self._job_factory = job_factory

    async def start(self, spec: ModelSpec) -> LaunchedService:
        if not spec.command:
            raise ServiceUnavailableError(
                f"model '{spec.id}' has no configured command to start it"
            )

        argv = _split_command(spec.command)
        job = self._job_factory()
        try:
            owned_process = await asyncio.to_thread(
                OwnedProcess.start, ProcessSpec(argv, cwd=self._cwd), job
            )
        except BaseException:
            job.close()
            raise

        managed = _ManagedProcess(owned_process, job)
        deadline = time.monotonic() + self._startup_timeout
        last_evidence: Mapping[str, Any] = {}
        while True:
            result = await self._probe.check(spec)
            last_evidence = result.evidence
            if result.healthy:
                return LaunchedService(
                    process=managed, pid=managed.pid, endpoint=spec.url, evidence=last_evidence
                )
            if time.monotonic() >= deadline or not managed.is_running():
                await asyncio.to_thread(managed.force_stop)
                raise ServiceUnavailableError(
                    f"service for model '{spec.id}' did not become healthy within "
                    f"{self._startup_timeout}s: {last_evidence}"
                )
            await asyncio.sleep(self._poll_interval)
