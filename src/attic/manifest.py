"""SQLite record of what is in the index, per file.

The manifest is the source of truth for "is this file indexed?". It is only ever updated
*after* the vector store write it describes has succeeded, so an interrupted sync leaves
the manifest describing an older, still-correct state and the next sync simply redoes the
unfinished files (see indexer.py for the full ordering argument).
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path       TEXT PRIMARY KEY,
    size       INTEGER NOT NULL,
    mtime_ns   INTEGER NOT NULL,
    sha256     TEXT NOT NULL,
    doc_id     TEXT,             -- NULL when nothing is stored for the file
    chunks     INTEGER NOT NULL,
    status     TEXT NOT NULL,    -- ok | empty | failed
    error      TEXT,
    indexed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass(frozen=True)
class Row:
    path: str
    size: int
    mtime_ns: int
    sha256: str
    doc_id: str | None
    chunks: int
    status: str
    error: str | None = None


class Manifest:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the watcher thread and the MCP server's handler share
        # one Manifest; every method below is a single statement or one transaction.
        self._db = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
        # WAL lets the search process read while the indexer writes.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def rows(self) -> dict[str, Row]:
        cur = self._db.execute(
            "SELECT path, size, mtime_ns, sha256, doc_id, chunks, status, error FROM files"
        )
        return {r[0]: Row(*r) for r in cur}

    def doc_ids(self) -> set[str]:
        cur = self._db.execute("SELECT doc_id FROM files WHERE doc_id IS NOT NULL")
        return {r[0] for r in cur}

    def upsert(self, rows: Iterable[Row]) -> None:
        now = time.time()
        with self._db:  # one transaction: the whole batch lands, or none of it
            self._db.executemany(
                "INSERT INTO files (path, size, mtime_ns, sha256, doc_id, chunks, status,"
                " error, indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(path) DO UPDATE SET size=excluded.size,"
                " mtime_ns=excluded.mtime_ns, sha256=excluded.sha256,"
                " doc_id=excluded.doc_id, chunks=excluded.chunks, status=excluded.status,"
                " error=excluded.error, indexed_at=excluded.indexed_at",
                [
                    (r.path, r.size, r.mtime_ns, r.sha256, r.doc_id, r.chunks, r.status,
                     r.error, now)
                    for r in rows
                ],
            )

    def remove(self, paths: Iterable[str]) -> None:
        with self._db:
            self._db.executemany("DELETE FROM files WHERE path = ?", [(p,) for p in paths])

    def get_meta(self, key: str, default: str = "") -> str:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def generation(self) -> int:
        """Bumped after every batch that changed the store. A searching process compares
        it to the value it last saw to know its view of the store is stale."""
        return int(self.get_meta("generation", "0"))

    def bump_generation(self) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO meta (key, value) VALUES ('generation', '1')"
                " ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"
            )

    def stats(self) -> dict[str, int]:
        cur = self._db.execute(
            "SELECT status, COUNT(*), COALESCE(SUM(chunks), 0) FROM files GROUP BY status"
        )
        out = {"ok": 0, "empty": 0, "failed": 0, "chunks": 0}
        for status, count, chunks in cur:
            out[status] = count
            out["chunks"] += chunks
        return out

    def failed(self, limit: int = 50) -> list[tuple[str, str]]:
        cur = self._db.execute(
            "SELECT path, error FROM files WHERE status = 'failed' ORDER BY path LIMIT ?",
            (limit,),
        )
        return [(r[0], r[1] or "") for r in cur]
