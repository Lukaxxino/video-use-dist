"""Small cross-platform, crash-safe file lock used by workspace stores."""

from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import time
from typing import Iterator, Type


_LOCK_CONTENTION_ERRNOS = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}


@contextmanager
def exclusive_file_lock(
    path: Path,
    *,
    timeout: float = 10.0,
    timeout_error: Type[Exception] = TimeoutError,
    timeout_message: str | None = None,
) -> Iterator[None]:
    """Hold an OS-managed one-byte exclusive lock until the context exits."""

    if timeout < 0:
        raise ValueError("lock timeout must be non-negative")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock_file:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()

        deadline = time.monotonic() + timeout
        while True:
            try:
                _try_lock_file(lock_file)
                break
            except OSError as error:
                if error.errno not in _LOCK_CONTENTION_ERRNOS:
                    raise
                if time.monotonic() >= deadline:
                    message = timeout_message or f"Timed out waiting for file lock: {path}"
                    raise timeout_error(message) from error
                time.sleep(0.01)

        try:
            yield
        finally:
            _unlock_file(lock_file)


def _try_lock_file(lock_file) -> None:
    lock_file.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(lock_file) -> None:
    lock_file.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
