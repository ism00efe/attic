"""Shared fixtures. Everything runs offline: a hashing fake stands in for the embedding
model, and Chroma runs for real in a temp directory (no model download, no GPU)."""

from __future__ import annotations

import hashlib
import math
import re

import pytest

from attic.config import Config

DIM = 64


class FakeEmbeddings:
    """Bag-of-words hashed into a fixed vector: texts sharing words land close together,
    which is enough to test that search finds the right file."""

    def __init__(self) -> None:
        self.document_calls = 0
        self.texts_embedded = 0
        self.fail_after: int | None = None  # raise once this many texts were embedded

    @staticmethod
    def _vec(text: str) -> list[float]:
        v = [0.0] * DIM
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail_after is not None and self.texts_embedded + len(texts) > self.fail_after:
            raise RuntimeError("simulated crash (CUDA out of memory)")
        self.document_calls += 1
        self.texts_embedded += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def fake() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "home"
    r.mkdir()
    return r


@pytest.fixture
def cfg(tmp_path, root) -> Config:
    return Config(
        roots=[str(root)],
        data_dir=str(tmp_path / "data"),
        batch_files=2,
        low_priority=False,
        chunk_size=200,
        chunk_overlap=20,
    )


@pytest.fixture
def make_indexer(cfg, fake):
    from attic.indexer import Indexer

    made = []

    def _make(**overrides):
        for key, value in overrides.items():
            setattr(cfg, key, value)
        idx = Indexer(cfg, embedder_factory=lambda _c: fake)
        made.append(idx)
        return idx

    yield _make
    for idx in made:
        idx.manifest.close()
