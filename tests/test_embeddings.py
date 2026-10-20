"""The real sentence-transformers wrapper, run against a tiny randomly initialised model
built on the spot (no download), plus the prompt formats."""

import math

import pytest

from attic.embeddings import (
    GEMMA,
    LocalEmbeddings,
    format_document,
    prompt_format_for,
    signature,
)


def test_gemma_prompts_and_signature():
    assert prompt_format_for("google/embeddinggemma-2") is GEMMA
    assert format_document("google/embeddinggemma-2", "oop/notes.md", "inheritance") == (
        "title: oop/notes.md | text: inheritance"
    )
    assert signature("google/embeddinggemma-2", None) != signature(
        "google/embeddinggemma-2", 256
    )


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    pytest.importorskip("torch")
    from sentence_transformers import SentenceTransformer, models
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "task", "search", "result", "query",
             "title", "text", "lentil", "soup", "java", "class", "|", ":"]
    tok = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]",
                                     pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]")
    bert = BertModel(BertConfig(vocab_size=len(words), hidden_size=32, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=64))
    raw = tmp_path_factory.mktemp("tiny-bert")
    bert.save_pretrained(raw)
    hf_tok.save_pretrained(raw)
    st = SentenceTransformer(modules=[models.Transformer(str(raw)),
                                      models.Pooling(32, pooling_mode="mean")])
    st.default_prompt_name = "poison"
    st.prompts = {"poison": "SHOULD NEVER BE PREPENDED "}
    out = tmp_path_factory.mktemp("tiny-st")
    st.save(str(out))
    return str(out)


def test_local_embeddings_normalizes_truncates_and_ignores_default_prompts(tiny_model_dir):
    emb = LocalEmbeddings(tiny_model_dir, device="cpu", dims=16)
    docs = emb.embed_documents(["title: a | text: lentil soup", "title: b | text: java class"])
    query = emb.embed_query("lentil soup")
    assert len(docs) == 2 and len(docs[0]) == 16 and len(query) == 16
    for v in (*docs, query):
        assert math.isclose(sum(x * x for x in v), 1.0, rel_tol=1e-4)

    # the model's default prompt must not be applied: same text, with and without it
    full = LocalEmbeddings(tiny_model_dir, device="cpu")
    plain = full._model.encode(["lentil soup"], prompt="")[0]
    ours = full.embed_documents(["lentil soup"])[0]
    norm = math.sqrt(sum(float(x) ** 2 for x in plain))
    pairs = zip(ours, plain, strict=True)
    assert all(math.isclose(a, float(b) / norm, abs_tol=1e-5) for a, b in pairs)
