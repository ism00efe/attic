"""attic configuration: one TOML file, every field optional.

Lookup order for the file: ``--config`` on the command line, then the ``ATTIC_CONFIG``
environment variable, then ``~/.attic/config.toml``.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

DEFAULT_HOME = Path("~/.attic")


@dataclass
class Config:
    # Folders to index. Nothing is indexed until at least one is listed.
    roots: list[str] = field(default_factory=list)
    # Where the vector store, manifest, lock and logs live.
    data_dir: str = str(DEFAULT_HOME)
    # Extra gitignore-style patterns on top of the built-in excludes (see ignore.py).
    exclude: list[str] = field(default_factory=list)
    # Set to false to also index dot-folders (.config, .obsidian, ...). Credential
    # folders and secret files stay excluded either way.
    skip_hidden: bool = True
    respect_gitignore: bool = True
    max_file_mb: float = 20.0
    # Mask e-mails, phone numbers, card numbers, tokens, IPs and Turkish ID numbers
    # (kuhaku's sanitizer) before indexing. Off by default: this is a private index of
    # your own files, and masking makes "what was X's phone number" unanswerable.
    mask_sensitive: bool = False

    # Changing the model or its vector size later means re-embedding every file.
    embedding_model: str = "google/embeddinggemma-2"
    embedding_dims: int | None = None  # None = the model's full size; else Matryoshka cut
    embedding_device: str = "auto"  # auto | cuda | cpu
    embedding_batch_size: int = 16
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Files embedded together per batch, and an optional pause between batches. The
    # pause is the knob for heat: it turns a long full-power run into a duty cycle.
    batch_files: int = 32
    pause_between_batches_seconds: float = 0.0
    low_priority: bool = True

    # watch mode: how long the folders must stay quiet before a sync runs.
    watch_debounce_seconds: float = 300.0

    retrieval: str = "hybrid"  # hybrid | dense

    @property
    def data_path(self) -> Path:
        return Path(os.path.expandvars(self.data_dir)).expanduser()

    @property
    def root_paths(self) -> list[Path]:
        return [Path(os.path.expandvars(r)).expanduser().resolve() for r in self.roots]

    @property
    def max_file_bytes(self) -> int:
        return int(self.max_file_mb * 1024 * 1024)


def default_config_path() -> Path:
    env = os.environ.get("ATTIC_CONFIG")
    if env:
        return Path(env).expanduser()
    return (DEFAULT_HOME / "config.toml").expanduser()


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path).expanduser() if path else default_config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No config at {cfg_path}. Run `attic init` to create one, then list your "
            "folders under `roots`."
        )
    with cfg_path.open("rb") as fh:
        raw = tomllib.load(fh)
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(raw) - known)
    if unknown:
        # A typo'd key silently doing nothing is worse than an error.
        raise ValueError(f"Unknown config keys in {cfg_path}: {', '.join(unknown)}")
    cfg = Config(**raw)
    if cfg.retrieval not in ("hybrid", "dense"):
        raise ValueError("retrieval must be 'hybrid' or 'dense'")
    if cfg.chunk_overlap >= cfg.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    return cfg


EXAMPLE_CONFIG = """\
# attic configuration. Every key is optional except `roots`.

# Folders to index. Use forward slashes or doubled backslashes on Windows.
roots = [
  "~/Documents",
  "~/Desktop",
]

# Extra gitignore-style patterns to skip, on top of the built-in ones
# (node_modules, .venv, .git, build outputs, caches, secrets, media, archives...).
exclude = [
  # "Downloads/old/",
  # "*.csv",
]

# data_dir = "~/.attic"
# skip_hidden = true
# respect_gitignore = true
# max_file_mb = 20

# Mask e-mails, phone/card/ID numbers, tokens and IPs before indexing.
# mask_sensitive = false

# Changing the model or embedding_dims later re-embeds every file (attic sync --rebuild).
# embedding_model = "google/embeddinggemma-2"
# embedding_dims = 768               # omit for the model's full size
# embedding_device = "auto"          # auto | cuda | cpu
# embedding_batch_size = 16

# Heat control for big runs: a pause after every batch keeps the GPU from running
# flat out. 0 = as fast as possible.
# batch_files = 32
# pause_between_batches_seconds = 0
# low_priority = true

# watch mode waits until folders have been quiet this long before syncing.
# watch_debounce_seconds = 300

# retrieval = "hybrid"               # hybrid (vector + keyword) | dense
"""
