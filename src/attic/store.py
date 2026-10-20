"""kuhaku's Chroma store, plus the two write operations incremental indexing needs.

kuhaku 0.2's ``VectorStore`` protocol only has ``add`` and ``reset``: enough for
"rebuild the corpus", not for "this one file changed". attic adds ``upsert`` and
``delete_documents`` here rather than in kuhaku. They go through
``ChromaVectorStore``'s own collection handle and metadata mapping, so stored rows keep
exactly kuhaku's shape and every kuhaku retriever reads them unchanged. That reaches into
two private members, which is why pyproject pins ``kuhaku<0.3`` and why construction
fails loudly if they are gone.
"""

from __future__ import annotations

from collections.abc import Iterable

from kuhaku.tools.rag import ChromaVectorStore, Chunk

COLLECTION = "attic"
_PAGE = 1000
_DELETE_BATCH = 200


class AtticStore(ChromaVectorStore):
    def __init__(self, persist_dir: str) -> None:
        super().__init__(persist_dir, COLLECTION)
        if not hasattr(self, "_collection") or not hasattr(self, "_to_stored_metadata"):
            raise RuntimeError(
                "This kuhaku version no longer exposes the Chroma collection attic "
                "relies on; attic supports kuhaku 0.2.x."
            )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        kwargs = {
            "ids": [c.id for c in chunks],
            "embeddings": embeddings,
            "documents": [c.text for c in chunks],
            "metadatas": [self._to_stored_metadata(c) for c in chunks],
        }
        try:
            self._collection.upsert(**kwargs)
        except Exception as exc:
            if "does not exist" not in str(exc).lower():
                raise
            self._refresh_collection_if_stale().upsert(**kwargs)

    def delete_documents(self, doc_ids: Iterable[str]) -> None:
        ids = list(dict.fromkeys(doc_ids))
        for start in range(0, len(ids), _DELETE_BATCH):
            batch = ids[start : start + _DELETE_BATCH]
            self._collection.delete(where={"document_id": {"$in": batch}})

    def document_ids(self) -> set[str]:
        """Every document id with at least one chunk stored. Pages through metadata only
        (no text, no vectors), so it stays cheap on a large index."""
        found: set[str] = set()
        offset = 0
        while True:
            page = self._collection.get(include=["metadatas"], limit=_PAGE, offset=offset)
            metas = page.get("metadatas") or []
            for meta in metas:
                if meta and meta.get("document_id"):
                    found.add(str(meta["document_id"]))
            if len(metas) < _PAGE:
                return found
            offset += _PAGE
