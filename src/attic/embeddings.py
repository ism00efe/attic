"""The embedding model and the exact text it sees.

Everything in this module is part of the index's "embedding signature": the model, the
vector size, and the prompt format that wraps each chunk and each query. Vectors made
under one signature are not comparable with another's, so a signature change is the one
change that needs a full `attic sync --rebuild` (see indexer.py). The formats below are
therefore versioned and must not be edited in place; add a new version instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("attic")


@dataclass(frozen=True)
class PromptFormat:
    name: str
    query: str  # "{text}" is replaced
    document: str  # "{title}" and "{text}" are replaced


# EmbeddingGemma's retrieval prompts, from its model card: the document side carries
# the title, which is how a chunk keeps its file name as context.
GEMMA = PromptFormat(
    name="gemma-retrieval-v1",
    query="task: search result | query: {text}",
    document="title: {title} | text: {text}",
)
E5 = PromptFormat(name="e5-v1", query="query: {text}", document="passage: {title}\n{text}")
QWEN3 = PromptFormat(
    name="qwen3-v1",
    query=(
        "Instruct: Given a search query, retrieve relevant passages from the user's "
        "files that answer it\nQuery:{text}"
    ),
    document="{title}\n{text}",
)
PLAIN = PromptFormat(name="plain-v1", query="{text}", document="{title}\n{text}")


def prompt_format_for(model_name: str) -> PromptFormat:
    lower = model_name.lower()
    if "embeddinggemma" in lower:
        return GEMMA
    if "e5" in lower:
        return E5
    if "qwen3-embedding" in lower:
        return QWEN3
    return PLAIN


def signature(model_name: str, dims: int | None) -> str:
    fmt = prompt_format_for(model_name)
    return f"{model_name}|dims={dims or 'full'}|{fmt.name}"


def format_document(model_name: str, title: str, text: str) -> str:
    return prompt_format_for(model_name).document.format(title=title, text=text)


class LocalEmbeddings:
    """sentence-transformers model behind kuhaku's ``EmbeddingProvider`` interface.

    ``embed_documents`` expects text already wrapped by :func:`format_document`;
    ``embed_query`` wraps the query itself. Prompts are always passed explicitly
    (``prompt=""``) so a default prompt in the model's own config can never be applied
    on top of ours.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        dims: int | None = None,
        batch_size: int = 16,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        logger.info("loading embedding model %s on %s", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        if device == "cuda" and "embeddinggemma" in model_name.lower():
            # Gemma activations overflow in float16 (NaNs or silently worse vectors);
            # bfloat16 halves memory without that problem. RTX 30xx/40xx support it.
            self._model.to(torch.bfloat16)
        self._format = prompt_format_for(model_name)
        self._dims = dims
        self._batch_size = batch_size

    def _encode(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        vectors = self._model.encode(
            texts,
            prompt="",
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            truncate_dim=self._dims,
        ).astype("float32")
        if not np.isfinite(vectors).all():
            raise RuntimeError("embedding model produced non-finite values")
        # Normalized after truncation: a truncated Matryoshka vector is no longer unit
        # length, and the store compares by cosine.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._format.query.format(text=text)])[0]
