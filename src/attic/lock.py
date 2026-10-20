"""Single-writer lock: only one process may change the index at a time.

An OS-level file lock rather than a "lock file exists" check, so a crashed process never
leaves a stale lock behind: the OS releases it when the process dies.
"""

from __future__ import annotations

import os
from pathlib import Path


class WriterBusy(RuntimeError):
    pass


class WriterLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")  # noqa: SIM115 - held open for the lock's lifetime
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            raise WriterBusy(
                "Another attic process is already indexing (a running `attic watch`, "
                "`attic serve --watch` or `attic sync`)."
            ) from exc
        self._fh = fh

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> WriterLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
