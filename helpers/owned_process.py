"""Lifecycle wrapper for subprocesses that belong to one explicit job."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Mapping, Optional, Sequence, Union


PathLike = Union[str, os.PathLike]


@dataclass(frozen=True)
class ProcessSpec:
    argv: Sequence[str]
    cwd: Optional[PathLike] = None
    env: Optional[Mapping[str, str]] = None

    def __post_init__(self):
        if not self.argv:
            raise ValueError("process argv must not be empty")
        object.__setattr__(self, "argv", tuple(os.fspath(argument) for argument in self.argv))
        if self.cwd is not None:
            object.__setattr__(self, "cwd", Path(self.cwd))
        if self.env is not None:
            object.__setattr__(self, "env", dict(self.env))


class OwnedProcess:
    """One process handle associated with a job owned by the current task."""

    def __init__(self, backend, job):
        self._backend = backend
        self._job = job
        self._lock = threading.RLock()

    @classmethod
    def start(cls, spec, job):
        if not isinstance(spec, ProcessSpec):
            raise TypeError("spec must be a ProcessSpec")
        return cls(job.start_assigned(spec), job)

    @property
    def pid(self):
        return self._backend.pid

    def is_running(self):
        with self._lock:
            return self._backend.is_running()

    def _force_exact_process_after_job_failure(self, job_error):
        process_errors = []
        try:
            self._backend.force_stop()
        except Exception as error:
            process_errors.append(error)
        try:
            self._backend.wait(1.0)
        except Exception as error:
            process_errors.append(error)
        try:
            if self._backend.is_running():
                process_errors.append(
                    RuntimeError("owned process is still running after termination")
                )
            else:
                self._backend.close()
        except Exception as error:
            process_errors.append(error)

        if process_errors:
            details = "; ".join(str(error) for error in process_errors)
            combined = RuntimeError(
                f"job termination failed: {job_error}; "
                f"owned process termination failed: {details}"
            )
            combined.job_error = job_error
            combined.process_errors = tuple(process_errors)
            raise combined from job_error
        raise job_error

    def terminate_gracefully(self, timeout):
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        with self._lock:
            if not self._backend.is_running():
                self._backend.close()
                return
            self._backend.send_graceful()
            try:
                self._backend.wait(timeout)
            except subprocess.TimeoutExpired:
                try:
                    self._job.terminate(exit_code=1)
                except Exception as job_error:
                    self._force_exact_process_after_job_failure(job_error)
                try:
                    self._backend.wait(min(max(timeout, 0.1), 5.0))
                except subprocess.TimeoutExpired:
                    self._backend.force_stop()
                    try:
                        self._backend.wait(1.0)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                active_error = sys.exc_info()[1]
                preserve_active_error = (
                    isinstance(active_error, RuntimeError)
                    and hasattr(active_error, "job_error")
                    and hasattr(active_error, "process_errors")
                )
                try:
                    if not self._backend.is_running():
                        self._backend.close()
                except Exception:
                    if not preserve_active_error:
                        raise

    def force_stop(self):
        with self._lock:
            if not self._backend.is_running():
                self._backend.close()
                return
            try:
                terminated_job = self._job.terminate(exit_code=1)
            except Exception as job_error:
                self._force_exact_process_after_job_failure(job_error)
            if terminated_job:
                try:
                    self._backend.wait(1.0)
                except subprocess.TimeoutExpired:
                    self._backend.force_stop()
            else:
                self._backend.force_stop()
            try:
                self._backend.wait(1.0)
            except subprocess.TimeoutExpired:
                pass
            finally:
                if not self._backend.is_running():
                    self._backend.close()
