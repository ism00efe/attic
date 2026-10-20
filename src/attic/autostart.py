"""Start `attic watch` automatically when you log in to Windows, and stop it on request.

Uses the per-user registry Run key (HKCU\\...\\CurrentVersion\\Run): no admin rights, and
it shows up under Task Manager > Startup apps, where it can also be switched off. The
watcher runs under pythonw, so no console window appears.

Autostart is switched on after the first successful `attic sync`, so the heavy first
index is always something you start yourself. `attic autostart off` removes it, stops
the running watcher and remembers the choice, so later syncs do not switch it back on.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from .manifest import Manifest

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "attic"
PID_FILE = "watch.pid"


class Backend(Protocol):
    def get(self) -> str | None: ...
    def set(self, command: str) -> None: ...
    def delete(self) -> None: ...


class WindowsRunKey:
    def get(self) -> str | None:
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
                return str(winreg.QueryValueEx(key, VALUE_NAME)[0])
        except FileNotFoundError:
            return None

    def set(self, command: str) -> None:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command)

    def delete(self) -> None:
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE
            ) as key:
                winreg.DeleteValue(key, VALUE_NAME)
        except FileNotFoundError:
            pass


def supported() -> bool:
    return os.name == "nt"


def watch_command(cfg_path: Path) -> list[str]:
    exe = Path(sys.executable)
    windowless = exe.with_name("pythonw.exe")
    interpreter = windowless if windowless.exists() else exe
    return [str(interpreter), "-m", "attic", "--config", str(cfg_path.resolve()), "watch"]


# ------------------------------------------------------------------ running watcher


def write_pid(data_dir: Path) -> None:
    (data_dir / PID_FILE).write_text(str(os.getpid()), encoding="ascii")


def clear_pid(data_dir: Path) -> None:
    path = data_dir / PID_FILE
    try:
        if path.read_text(encoding="ascii").strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass


def running_pid(data_dir: Path) -> int | None:
    """The pid of a live `attic watch`, checked against the process's command line so a
    stale pid file pointing at an unrelated, reused pid is never acted on."""
    import psutil

    try:
        pid = int((data_dir / PID_FILE).read_text(encoding="ascii").strip())
        cmdline = " ".join(psutil.Process(pid).cmdline())
    except (OSError, ValueError, psutil.Error):
        return None
    return pid if "attic" in cmdline and "watch" in cmdline else None


def stop(data_dir: Path) -> bool:
    import psutil

    pid = running_pid(data_dir)
    if pid is None:
        return False
    proc = psutil.Process(pid)
    proc.terminate()  # an interrupted sync is safe to resume (see indexer.py)
    try:
        proc.wait(timeout=15)
    except psutil.TimeoutExpired:
        proc.kill()
    try:
        (data_dir / PID_FILE).unlink()
    except OSError:
        pass
    return True


def start_detached(command: list[str]) -> None:
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, close_fds=True, creationflags=flags,
        start_new_session=os.name != "nt",
    )


# ------------------------------------------------------------------ on / off


class Autostart:
    def __init__(self, cfg_path: Path, data_dir: Path, manifest: Manifest,
                 backend: Backend | None = None) -> None:
        self.cfg_path = cfg_path
        self.data_dir = data_dir
        self.manifest = manifest
        self.backend = backend if backend is not None else WindowsRunKey()

    def enabled(self) -> bool:
        return self.backend.get() is not None

    def opted_out(self) -> bool:
        return self.manifest.get_meta("autostart") == "off"

    def enable(self, *, start_now: bool = True) -> None:
        command = watch_command(self.cfg_path)
        self.backend.set(subprocess.list2cmdline(command))
        self.manifest.set_meta("autostart", "on")
        if start_now and running_pid(self.data_dir) is None:
            start_detached(command)

    def disable(self) -> bool:
        """Remove from logon and stop the running watcher. Returns whether one was
        running."""
        self.backend.delete()
        self.manifest.set_meta("autostart", "off")
        return stop(self.data_dir)

    def enable_after_first_sync(self) -> bool:
        if not supported() or self.opted_out() or self.enabled():
            return False
        self.enable()
        return True
