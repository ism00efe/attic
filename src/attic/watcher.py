"""Watch mode: sync once at start, then again whenever the roots have changed and gone
quiet for ``watch_debounce_seconds``.

File events only *mark* the index dirty; the sync they trigger re-scans and decides what
actually changed. So a missed or duplicated event can never corrupt anything: at worst a
change waits for the next event, and a restart's initial sync catches everything.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .indexer import Indexer
from .lock import WriterBusy

logger = logging.getLogger("attic")


class _Handler(FileSystemEventHandler):
    def __init__(self, watcher: Watcher) -> None:
        self._watcher = watcher

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type in ("opened", "closed", "closed_no_write"):
            return
        # A folder's "modified" event only echoes a change to a file inside it, which
        # arrives as its own event; reacting to both would let `npm install` in an
        # ignored node_modules wake the indexer.
        if event.is_directory and event.event_type == "modified":
            return
        walker = self._watcher.indexer.walker
        for raw in (event.src_path, getattr(event, "dest_path", "")):
            if not raw:
                continue
            path = Path(str(raw))
            wanted = walker.is_wanted_dir(path) if event.is_directory else walker.is_wanted(path)
            if wanted:
                self._watcher.mark_dirty()
                return


class Watcher:
    def __init__(self, indexer: Indexer, *, debounce_seconds: float) -> None:
        self.indexer = indexer
        self.debounce = debounce_seconds
        self._dirty = threading.Event()
        self._last_event = 0.0
        self._lock = threading.Lock()

    def mark_dirty(self) -> None:
        with self._lock:
            self._last_event = time.monotonic()
        self._dirty.set()

    def _quiet_for(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_event

    def _sync(self) -> None:
        try:
            self.indexer.sync()
        except WriterBusy as exc:
            logger.warning("watch: %s Will retry after the next change.", exc)
        except Exception:
            logger.exception("watch: sync failed; will retry after the next change")

    def run(self, stop: threading.Event) -> None:
        observer = Observer()
        handler = _Handler(self)
        for root in self.indexer.cfg.root_paths:
            if root.is_dir():
                observer.schedule(handler, str(root), recursive=True)
        observer.start()
        try:
            self._sync()
            while not stop.is_set():
                if not self._dirty.wait(timeout=1.0):
                    continue
                remaining = self.debounce - self._quiet_for()
                if remaining > 0:
                    stop.wait(min(remaining, 5.0))
                    continue
                self._dirty.clear()
                self._sync()
        finally:
            observer.stop()
            observer.join(timeout=5)
