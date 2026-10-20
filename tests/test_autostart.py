import subprocess
import sys
import time

import pytest

from attic import autostart
from attic.autostart import Autostart, running_pid, stop
from attic.manifest import Manifest


class FakeRunKey:
    def __init__(self):
        self.value = None

    def get(self):
        return self.value

    def set(self, command):
        self.value = command

    def delete(self):
        self.value = None


@pytest.fixture
def auto(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(autostart, "start_detached", started.append)
    monkeypatch.setattr(autostart, "supported", lambda: True)
    manifest = Manifest(tmp_path / "manifest.db")
    a = Autostart(tmp_path / "config.toml", tmp_path, manifest, backend=FakeRunKey())
    a.started = started
    yield a
    manifest.close()


def test_first_sync_turns_autostart_on_and_starts_the_watcher(auto):
    assert auto.enable_after_first_sync()
    assert auto.enabled()
    assert auto.backend.value.endswith("watch") and "--config" in auto.backend.value
    assert len(auto.started) == 1
    assert not auto.enable_after_first_sync()  # already on: no second watcher


def test_turning_it_off_is_remembered(auto):
    auto.enable_after_first_sync()
    auto.disable()
    assert not auto.enabled() and auto.opted_out()
    assert not auto.enable_after_first_sync()  # later syncs leave the choice alone
    auto.enable()
    assert auto.enabled() and not auto.opted_out()


def test_not_supported_platform_is_left_alone(auto, monkeypatch):
    monkeypatch.setattr(autostart, "supported", lambda: False)
    assert not auto.enable_after_first_sync()
    assert not auto.enabled()


def test_stop_ends_a_running_watcher_and_ignores_unrelated_pids(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                             "attic", "watch"])
    try:
        (tmp_path / "watch.pid").write_text(str(proc.pid))
        deadline = time.monotonic() + 10
        while running_pid(tmp_path) is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert running_pid(tmp_path) == proc.pid
        assert stop(tmp_path)
        assert proc.wait(timeout=15) is not None
        assert not (tmp_path / "watch.pid").exists()
    finally:
        proc.kill()

    # a pid file pointing at some other process is never acted on
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (tmp_path / "watch.pid").write_text(str(other.pid))
        assert running_pid(tmp_path) is None
        assert not stop(tmp_path)
        assert other.poll() is None
    finally:
        other.kill()
