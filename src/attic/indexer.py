"""Incremental, interruption-safe sync of the roots into the vector store.

Ordering, and why an interrupted sync is always safe to re-run:

1. New chunks are written under a *new* document id (path key + content hash), so they
   never overwrite the file's previous chunks.
2. Only then is the manifest updated to point at the new id.
3. Only then are the previous version's chunks deleted.

Stopped after (1): the new chunks are orphans (no manifest row names them) and the
manifest still describes the old file, so the next sync deletes the orphans and redoes
the file. Stopped after (2): the old chunks are the orphans and get deleted. At no point
is a file absent from search because of a crash. Every sync begins with that orphan
sweep, so leftovers never accumulate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from kuhaku.core.sanitization import sanitize
from kuhaku.tools.rag import Chunk, Document, EmbeddingProvider, chunk_document

from .config import Config
from .ignore import FileInfo, Walker
from .loaders import LoadError, load_text
from .lock import WriterLock
from .manifest import Manifest, Row
from .store import AtticStore

logger = logging.getLogger("attic")

EmbedderFactory = Callable[[Config], EmbeddingProvider]


class SettingsChanged(RuntimeError):
    pass


def _index_settings(cfg: Config) -> dict[str, object]:
    """Settings that shape what is stored. Changing any of them makes the existing
    index inconsistent with new writes (vectors from two models are not comparable;
    old chunks keep their old size or masking), so a change requires a rebuild."""
    return {
        "embedding_model": cfg.embedding_model,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "mask_sensitive": cfg.mask_sensitive,
    }


@dataclass
class Plan:
    new: list[FileInfo] = field(default_factory=list)
    changed: list[FileInfo] = field(default_factory=list)
    deleted: list[Row] = field(default_factory=list)
    unchanged: int = 0
    # Paths whose chunks are missing from the store: re-embedded even if unchanged.
    forced: set[str] = field(default_factory=set)

    @property
    def to_process(self) -> int:
        return len(self.new) + len(self.changed)


@dataclass
class Report:
    plan_new: int = 0
    plan_changed: int = 0
    plan_deleted: int = 0
    unchanged: int = 0
    indexed: int = 0
    content_unchanged: int = 0  # mtime moved but the bytes did not: no re-embedding
    empty: int = 0
    failed: int = 0
    skipped_busy: int = 0  # locked/unreadable right now (e.g. open in Word): retried later
    removed: int = 0
    orphans_removed: int = 0
    chunks_written: int = 0
    seconds: float = 0.0
    dry_run: bool = False

    def summary(self) -> str:
        if self.dry_run:
            return (
                f"would index {self.plan_new} new and {self.plan_changed} changed files, "
                f"remove {self.plan_deleted}; {self.unchanged} unchanged"
            )
        return (
            f"indexed {self.indexed} files ({self.chunks_written} chunks), "
            f"removed {self.removed}, unchanged {self.unchanged + self.content_unchanged}, "
            f"empty {self.empty}, failed {self.failed}, busy {self.skipped_busy}, "
            f"orphans cleaned {self.orphans_removed} in {self.seconds:.1f}s"
        )


def default_embedder(cfg: Config) -> EmbeddingProvider:
    from kuhaku.core.resolve.probes import torch_accelerator
    from kuhaku.tools.rag import SentenceTransformerEmbeddings

    device = cfg.embedding_device if cfg.embedding_device != "auto" else torch_accelerator()
    logger.info("loading embedding model %s on %s", cfg.embedding_model, device)
    return SentenceTransformerEmbeddings(cfg.embedding_model, device=device)


def path_key(path: str) -> str:
    return hashlib.sha1(os.path.normcase(path).encode("utf-8")).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _doc_type(suffix: str) -> str:
    if suffix in (".pdf", ".docx", ".pptx", ".xlsx"):
        return suffix.lstrip(".")
    if suffix in (".md", ".markdown", ".txt", ".rst", ".org", ".tex"):
        return "note"
    if suffix in (".csv", ".tsv", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
                  ".conf", ".xml"):
        return "data"
    if suffix in (".html", ".htm"):
        return "web"
    return "code"


class Indexer:
    def __init__(
        self,
        cfg: Config,
        *,
        embedder_factory: EmbedderFactory = default_embedder,
        store: AtticStore | None = None,
        manifest: Manifest | None = None,
    ) -> None:
        self.cfg = cfg
        data = cfg.data_path
        data.mkdir(parents=True, exist_ok=True)
        self.store = store or AtticStore(str(data / "chroma"))
        self.manifest = manifest or Manifest(data / "manifest.db")
        self.lock = WriterLock(data / "writer.lock")
        self.walker = Walker(
            cfg.root_paths,
            exclude=cfg.exclude,
            skip_hidden=cfg.skip_hidden,
            respect_gitignore=cfg.respect_gitignore,
            max_file_bytes=cfg.max_file_bytes,
        )
        self._embedder_factory = embedder_factory
        self._embedder: EmbeddingProvider | None = None

    @property
    def embedder(self) -> EmbeddingProvider:
        # Lazy: a sync that finds nothing to embed never loads the model (no torch import,
        # no GPU wake-up). That is what keeps routine syncs cheap.
        if self._embedder is None:
            self._embedder = self._embedder_factory(self.cfg)
        return self._embedder

    # ---------------------------------------------------------------- planning

    def plan(self) -> Plan:
        rows = self.manifest.rows()
        plan = Plan()
        seen: set[str] = set()
        for info in self.walker.walk():
            key = str(info.path)
            seen.add(key)
            row = rows.get(key)
            if row is None:
                plan.new.append(info)
            elif row.size != info.size or row.mtime_ns != info.mtime_ns:
                plan.changed.append(info)
            else:
                plan.unchanged += 1
        plan.deleted = [row for path, row in rows.items() if path not in seen]
        return plan

    # ---------------------------------------------------------------- sync

    def sync(
        self,
        *,
        dry_run: bool = False,
        rebuild: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> Report:
        started = time.monotonic()
        with self.lock:
            if self.cfg.low_priority and not dry_run:
                _lower_priority()
            self._check_settings(rebuild and not dry_run)
            report = Report(dry_run=dry_run)
            plan = self.plan()
            if not dry_run:
                stored = self.store.document_ids()
                report.orphans_removed = self._sweep_orphans(stored)
                self._requeue_missing(plan, stored)
            report.plan_new = len(plan.new)
            report.plan_changed = len(plan.changed)
            report.plan_deleted = len(plan.deleted)
            report.unchanged = plan.unchanged
            if dry_run:
                return report

            self._remove(plan.deleted, report)
            todo = plan.new + plan.changed
            rows = self.manifest.rows()
            for start in range(0, len(todo), self.cfg.batch_files):
                batch = todo[start : start + self.cfg.batch_files]
                self._index_batch(batch, rows, plan.forced, report)
                if progress:
                    progress(min(start + len(batch), len(todo)), len(todo))
                if self.cfg.pause_between_batches_seconds > 0 and start + len(batch) < len(todo):
                    time.sleep(self.cfg.pause_between_batches_seconds)

            self.manifest.set_meta("last_sync_at", str(time.time()))
            report.seconds = time.monotonic() - started
            logger.info("sync: %s", report.summary())
            return report

    def _check_settings(self, rebuild: bool) -> None:
        current = _index_settings(self.cfg)
        raw = self.manifest.get_meta("index_settings")
        stored = json.loads(raw) if raw else None
        if rebuild:
            self.store.reset()
            self.manifest.remove(self.manifest.rows().keys())
            self.manifest.bump_generation()
        elif stored is not None and stored != current:
            changed = ", ".join(
                f"{k} {stored.get(k)!r} -> {v!r}" for k, v in current.items()
                if stored.get(k) != v
            )
            raise SettingsChanged(
                f"Index settings changed since the index was built ({changed}). Run "
                "`attic sync --rebuild` to re-index everything with the new settings."
            )
        self.manifest.set_meta("index_settings", json.dumps(current, sort_keys=True))
        self.manifest.set_meta("embedding_model", self.cfg.embedding_model)

    def _sweep_orphans(self, stored: set[str]) -> int:
        orphans = stored - self.manifest.doc_ids()
        if orphans:
            self.store.delete_documents(orphans)
            self.manifest.bump_generation()
        return len(orphans)

    def _requeue_missing(self, plan: Plan, stored: set[str]) -> None:
        """Files the manifest says are indexed but whose chunks are gone (the store was
        deleted or restored from an older copy) are re-embedded instead of silently
        staying unsearchable."""
        queued = {str(i.path) for i in plan.new + plan.changed}
        deleted = {r.path for r in plan.deleted}
        for path, row in self.manifest.rows().items():
            if row.status != "ok" or row.doc_id in stored or path in queued or path in deleted:
                continue
            plan.changed.append(FileInfo(path=Path(path), size=row.size, mtime_ns=row.mtime_ns))
            plan.forced.add(path)
            plan.unchanged -= 1

    def _remove(self, deleted: list[Row], report: Report) -> None:
        if not deleted:
            return
        self.store.delete_documents(r.doc_id for r in deleted if r.doc_id)
        self.manifest.remove(r.path for r in deleted)
        self.manifest.bump_generation()
        report.removed = len(deleted)

    def _index_batch(
        self, batch: list[FileInfo], rows: dict[str, Row], forced: set[str], report: Report
    ) -> None:
        new_rows: list[Row] = []
        chunks: list[Chunk] = []
        embed_texts: list[str] = []
        stale_doc_ids: list[str] = []

        for info in batch:
            key = str(info.path)
            old = rows.get(key)
            try:
                sha = file_sha256(info.path)
            except OSError:
                report.skipped_busy += 1
                continue
            if old is not None and old.sha256 == sha and key not in forced:
                # Same bytes, new timestamp (copy, cloud-sync tool, `touch`): only the
                # manifest's size/mtime move; nothing is re-read or re-embedded. A file
                # that failed to parse stays failed until its bytes change.
                new_rows.append(Row(key, info.size, info.mtime_ns, sha, old.doc_id,
                                    old.chunks, old.status, old.error))
                report.content_unchanged += 1
                continue
            doc_id = f"{path_key(key)}-{sha[:16]}"
            try:
                text = load_text(info.path)
            except LoadError as exc:
                new_rows.append(Row(key, info.size, info.mtime_ns, sha, None, 0, "failed",
                                    str(exc)[:300]))
                report.failed += 1
                if old and old.doc_id:
                    stale_doc_ids.append(old.doc_id)
                continue
            except OSError:
                report.skipped_busy += 1
                continue
            if self.cfg.mask_sensitive:
                text = sanitize(text)
            text = text.strip()
            file_chunks = self._chunk(info.path, doc_id, text) if text else []
            if not file_chunks:
                new_rows.append(Row(key, info.size, info.mtime_ns, sha, None, 0, "empty"))
                report.empty += 1
                if old and old.doc_id:
                    stale_doc_ids.append(old.doc_id)
                continue
            header = f"File: {info.path.parent.name}/{info.path.name}\n"
            chunks.extend(file_chunks)
            embed_texts.extend(header + c.text for c in file_chunks)
            new_rows.append(Row(key, info.size, info.mtime_ns, sha, doc_id,
                                len(file_chunks), "ok"))
            if old and old.doc_id and old.doc_id != doc_id:
                stale_doc_ids.append(old.doc_id)
            report.indexed += 1

        if chunks:
            vectors = self.embedder.embed_documents(embed_texts)
            self.store.upsert(chunks, vectors)  # step 1: new version stored
            report.chunks_written += len(chunks)
        if new_rows:
            self.manifest.upsert(new_rows)  # step 2: manifest points at it
        if stale_doc_ids:
            self.store.delete_documents(stale_doc_ids)  # step 3: old version removed
        if chunks or stale_doc_ids:
            self.manifest.bump_generation()

    def _chunk(self, path: Path, doc_id: str, text: str) -> list[Chunk]:
        doc = Document(
            id=doc_id,
            title=path.name,
            doc_type=_doc_type(path.suffix.lower()),
            text=text,
            source_path=str(path),
        )
        return chunk_document(doc, self.cfg.chunk_size, self.cfg.chunk_overlap)


def _lower_priority() -> None:
    try:
        import psutil

        proc = psutil.Process()
        if os.name == "nt":
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            proc.nice(10)
    except Exception:  # best effort: a priority we cannot set is not worth failing a sync
        pass
