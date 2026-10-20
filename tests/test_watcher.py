import threading
import time

from attic.watcher import Watcher


def _wait_for(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def test_watch_syncs_at_start_and_after_changes_settle(root, make_indexer):
    (root / "first.md").write_text("first note", encoding="utf-8")
    idx = make_indexer()
    stop = threading.Event()
    thread = threading.Thread(target=Watcher(idx, debounce_seconds=0.5).run, args=(stop,))
    thread.start()
    try:
        assert _wait_for(lambda: idx.store.count() == 1)  # initial sync
        (root / "second.md").write_text("second note", encoding="utf-8")
        assert _wait_for(lambda: idx.store.count() == 2)
        (root / "first.md").unlink()
        assert _wait_for(lambda: idx.store.count() == 1)
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not thread.is_alive()


def test_events_in_ignored_places_do_not_trigger_a_sync(root, make_indexer):
    idx = make_indexer()
    watcher = Watcher(idx, debounce_seconds=0.1)
    from watchdog.events import DirCreatedEvent, DirModifiedEvent, FileCreatedEvent

    from attic.watcher import _Handler

    handler = _Handler(watcher)
    handler.on_any_event(FileCreatedEvent(str(root / "node_modules" / "x.js")))
    handler.on_any_event(FileCreatedEvent(str(root / "clip.mp4")))
    handler.on_any_event(DirCreatedEvent(str(root / "app" / "node_modules" / "pkg")))
    handler.on_any_event(DirModifiedEvent(str(root / "notes")))
    assert not watcher._dirty.is_set()
    handler.on_any_event(DirCreatedEvent(str(root / "new-project")))
    assert watcher._dirty.is_set()
    watcher._dirty.clear()
    handler.on_any_event(FileCreatedEvent(str(root / "note.md")))
    assert watcher._dirty.is_set()
