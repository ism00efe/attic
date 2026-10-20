"""Query side: kuhaku's retrievers over the attic store.

The searching process is often not the indexing one (Claude Desktop starts `attic serve`
while a scheduled `attic sync` runs separately). Chroma keeps an in-memory vector index
per process that does not see another process's writes, so the searcher watches the
manifest's generation counter and reopens the store when it moves.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass

from kuhaku.tools.rag import (
    DenseRetriever,
    EmbeddingProvider,
    HybridRetriever,
    StoreBackedBM25Retriever,
)

from .config import Config
from .indexer import default_embedder
from .manifest import Manifest
from .store import AtticStore

logger = logging.getLogger("attic")


@dataclass
class Hit:
    path: str
    file: str
    score: float
    chunk: int
    text: str

    def as_dict(self) -> dict:
        return asdict(self)


class Searcher:
    def __init__(
        self,
        cfg: Config,
        *,
        embedder_factory: Callable[[Config], EmbeddingProvider] = default_embedder,
        store: AtticStore | None = None,
        manifest: Manifest | None = None,
    ) -> None:
        self.cfg = cfg
        data = cfg.data_path
        self.manifest = manifest or Manifest(data / "manifest.db")
        # A store handed in is shared with an in-process indexer: its writes are already
        # visible here, and reopening it would pull the client out from under the writer.
        self._shared_store = store is not None
        self._store = store
        self._embedder_factory = embedder_factory
        self._embedder: EmbeddingProvider | None = None
        self._retriever = None
        self._generation = -1
        self._lock = threading.Lock()

    @property
    def embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            self._embedder = self._embedder_factory(self.cfg)
        return self._embedder

    def _store_now(self) -> AtticStore:
        generation = self.manifest.generation()
        if self._store is None or generation != self._generation:
            if self._store is not None and not self._shared_store:
                from chromadb.api.client import SharedSystemClient

                SharedSystemClient.clear_system_cache()
                self._store = None
            if self._store is None:
                self._store = AtticStore(str(self.cfg.data_path / "chroma"))
            self._retriever = None
            self._generation = generation
        return self._store

    def _retriever_now(self):
        store = self._store_now()
        if self._retriever is None:
            dense = DenseRetriever(self.embedder, store)
            if self.cfg.retrieval == "dense":
                self._retriever = dense
            else:
                self._retriever = HybridRetriever(
                    dense, StoreBackedBM25Retriever(store), candidates=60,
                    max_chunks_per_document=2,
                )
        return self._retriever

    def search(self, query: str, top_k: int = 8, path_contains: str | None = None) -> list[Hit]:
        query = query.strip()
        if not query:
            return []
        top_k = max(1, min(top_k, 50))
        fetch = top_k * 5 if path_contains else top_k
        with self._lock:
            results = self._retriever_now().retrieve(query, fetch)
        hits = []
        needle = path_contains.lower() if path_contains else None
        for r in results:
            path = r.chunk.source_path
            if needle and needle not in path.lower():
                continue
            hits.append(Hit(path=path, file=r.chunk.title, score=round(float(r.score), 4),
                            chunk=r.chunk.chunk_index, text=r.chunk.text))
            if len(hits) == top_k:
                break
        return hits

    def document_text(self, path: str, max_chars: int = 20000) -> str | None:
        """The indexed text of one file, reassembled from its chunks. Only what is in the
        index: this never reads arbitrary paths from disk."""
        row = self.manifest.rows().get(path)
        if row is None or row.doc_id is None:
            return None
        with self._lock:
            chunks = self._store_now().get_by_document_id(row.doc_id)
        chunks.sort(key=lambda c: c.chunk_index)
        text = "\n\n".join(c.text for c in chunks)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[truncated: {len(text) - max_chars} more chars]"
        return text

    def status(self) -> dict:
        stats = self.manifest.stats()
        last = self.manifest.get_meta("last_sync_at")
        return {
            "files_indexed": stats["ok"],
            "files_empty": stats["empty"],
            "files_failed": stats["failed"],
            "chunks": stats["chunks"],
            "last_sync_at": float(last) if last else None,
            "embedding_model": self.manifest.get_meta("embedding_model") or None,
            "roots": [str(p) for p in self.cfg.root_paths],
        }
