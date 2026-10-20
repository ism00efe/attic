"""Which files under the roots get indexed.

The walk prunes whole directories as early as possible: on a real home folder almost all
of the cost is in trees nobody wants indexed (node_modules, virtualenvs, caches), and
pruning them means they are never even listed, let alone hashed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pathspec

from .loaders import SUPPORTED_SUFFIXES

# Build output, dependency trees and tool caches: regenerable, huge, and noise in search.
_SKIP_DIR_NAMES = {
    "node_modules", "__pycache__", "site-packages", "dist", "build", "target", "obj",
    ".git", ".hg", ".svn", ".venv", "venv", ".tox", ".nox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", ".gradle", ".next", ".nuxt",
    ".idea", ".vs", ".cache", "$recycle.bin", "system volume information", "appdata",
}

# Credential stores. Excluded even with skip_hidden = false: searching them is never the
# point, and an index is one more copy of their contents. No switch, because it costs
# nothing.
_SECRET_DIR_NAMES = {
    ".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".password-store",
    ".mcp-auth", ".kaggle",
}
_SECRET_FILE_PATTERNS = pathspec.GitIgnoreSpec.from_lines([
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "*.kdbx", "id_rsa*",
    "id_ed25519*", "id_ecdsa*", "credentials*.json", "*secret*", ".netrc", ".pgpass",
    ".npmrc", ".pypirc",
])


@dataclass(frozen=True)
class FileInfo:
    path: Path
    size: int
    mtime_ns: int


class Walker:
    def __init__(
        self,
        roots: list[Path],
        *,
        exclude: list[str] | None = None,
        skip_hidden: bool = True,
        respect_gitignore: bool = True,
        max_file_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        self.roots = roots
        self.user_spec = pathspec.GitIgnoreSpec.from_lines(exclude or [])
        self.skip_hidden = skip_hidden
        self.respect_gitignore = respect_gitignore
        self.max_file_bytes = max_file_bytes

    def walk(self) -> Iterator[FileInfo]:
        seen_roots: set[str] = set()
        for root in self.roots:
            if not root.is_dir():
                continue
            key = os.path.normcase(str(root))
            if key in seen_roots:
                continue
            seen_roots.add(key)
            yield from self._walk_root(root)

    def is_wanted(self, path: Path) -> bool:
        """Cheap check used by the file watcher to drop irrelevant events. Does not
        consult .gitignore files (that would mean re-reading them per event); the sync
        that the event triggers applies the full rules."""
        root = self._root_of(path)
        if root is None:
            return False
        rel = path.relative_to(root)
        for part in rel.parts[:-1]:
            if self._dir_excluded_by_name(part):
                return False
        return self._file_name_ok(path.name) and not self.user_spec.match_file(rel.as_posix())

    def is_wanted_dir(self, path: Path) -> bool:
        """Same idea for directory events (create/delete/move of a whole folder)."""
        root = self._root_of(path)
        if root is None:
            return False
        rel = path.relative_to(root)
        if any(self._dir_excluded_by_name(part) for part in rel.parts):
            return False
        return not rel.parts or not self.user_spec.match_file(rel.as_posix() + "/")

    def _root_of(self, path: Path) -> Path | None:
        for root in self.roots:
            try:
                path.relative_to(root)
            except ValueError:
                continue
            return root
        return None

    def _dir_excluded_by_name(self, name: str) -> bool:
        lower = name.lower()
        if lower in _SECRET_DIR_NAMES or lower in _SKIP_DIR_NAMES:
            return True
        return self.skip_hidden and name.startswith(".")

    def _file_name_ok(self, name: str) -> bool:
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            return False
        if _SECRET_FILE_PATTERNS.match_file(name):
            return False
        return not (self.skip_hidden and name.startswith("."))

    def _walk_root(self, root: Path) -> Iterator[FileInfo]:
        # Each stack entry carries the .gitignore specs in force for that directory,
        # as (directory the spec is relative to, spec).
        stack: list[tuple[Path, list[tuple[Path, pathspec.PathSpec]]]] = [(root, [])]
        while stack:
            directory, specs = stack.pop()
            try:
                entries = list(os.scandir(directory))
            except OSError:
                continue  # permission denied, vanished mid-walk: skip, never abort the sync
            if self.respect_gitignore and any(e.name == ".gitignore" for e in entries):
                try:
                    lines = (directory / ".gitignore").read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    specs = [*specs, (directory, pathspec.GitIgnoreSpec.from_lines(lines))]
                except OSError:
                    pass
            if any(e.name == "pyvenv.cfg" for e in entries) and directory != root:
                continue  # a virtualenv under any name
            for entry in entries:
                path = Path(entry.path)
                try:
                    if entry.is_symlink() or _is_junction(entry):
                        continue  # avoids cycles and double-indexing
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                rel = path.relative_to(root).as_posix()
                if is_dir:
                    if self._dir_excluded_by_name(entry.name):
                        continue
                    if self.user_spec.match_file(rel + "/") or _ignored(path, specs, True):
                        continue
                    stack.append((path, specs))
                    continue
                if not self._file_name_ok(entry.name):
                    continue
                if self.user_spec.match_file(rel) or _ignored(path, specs, False):
                    continue
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if st.st_size == 0 or st.st_size > self.max_file_bytes:
                    continue
                yield FileInfo(path=path, size=st.st_size, mtime_ns=st.st_mtime_ns)


def _ignored(path: Path, specs: list[tuple[Path, pathspec.PathSpec]], is_dir: bool) -> bool:
    for base, spec in specs:
        rel = path.relative_to(base).as_posix()
        if spec.match_file(rel + "/" if is_dir else rel):
            return True
    return False


def _is_junction(entry: os.DirEntry[str]) -> bool:
    is_junction = getattr(entry, "is_junction", None)  # Python 3.12+
    return bool(is_junction and is_junction())
