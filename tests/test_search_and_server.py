import asyncio
import os
import subprocess
import sys
import textwrap

from attic.search import Searcher
from attic.server import build_server


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _index(root, make_indexer):
    _write(root, "school/oop/notes.md", "java inheritance polymorphism interface")
    _write(root, "projects/kuhaku/router.py", "class Router:\n    def route(self): pass")
    _write(root, "personal/recipes.txt", "lentil soup with lemon")
    return make_indexer().sync()


def test_search_finds_the_right_file(root, cfg, fake, make_indexer):
    _index(root, make_indexer)
    for retrieval in ("hybrid", "dense"):
        cfg.retrieval = retrieval
        hits = Searcher(cfg, embedder_factory=lambda _c: fake).search("lentil soup", top_k=1)
        assert hits[0].path.endswith("recipes.txt"), retrieval


def test_path_filter(root, cfg, fake, make_indexer):
    _index(root, make_indexer)
    s = Searcher(cfg, embedder_factory=lambda _c: fake)
    hits = s.search("class route", top_k=5, path_contains="KUHAKU")
    assert hits and all("kuhaku" in h.path for h in hits)


def test_document_text_only_serves_indexed_files(root, cfg, fake, make_indexer):
    _index(root, make_indexer)
    s = Searcher(cfg, embedder_factory=lambda _c: fake)
    path = str(root / "personal" / "recipes.txt")
    assert s.document_text(path) == "lentil soup with lemon"
    assert s.document_text(str(root / "not" / "indexed.txt")) is None


def test_searcher_sees_changes_made_by_another_process(root, cfg, fake, make_indexer, tmp_path):
    """A long-running server must pick up a separate `attic sync`'s writes: Chroma's
    per-process vector index does not, without the generation-based reopen."""
    _index(root, make_indexer)
    s = Searcher(cfg, embedder_factory=lambda _c: fake)
    assert not s.search("zebra migration", top_k=3, path_contains="zebra")

    _write(root, "zebra.md", "zebra migration serengeti")
    script = tmp_path / "other_process.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {os.getcwd()!r})
        from tests.conftest import FakeEmbeddings
        from attic.config import Config
        from attic.indexer import Indexer
        cfg = Config(roots=[{str(root)!r}], data_dir={cfg.data_dir!r}, low_priority=False,
                     chunk_size=200, chunk_overlap=20)
        print(Indexer(cfg, embedder_factory=lambda c: FakeEmbeddings()).sync().indexed)
    """))
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                         timeout=120)
    assert out.stdout.strip() == "1", out.stderr[-500:]

    hits = s.search("zebra migration", top_k=3, path_contains="zebra")
    assert hits and hits[0].path.endswith("zebra.md")


def test_mcp_tools(root, cfg, fake, make_indexer):
    _index(root, make_indexer)
    server = build_server(cfg, Searcher(cfg, embedder_factory=lambda _c: fake))

    async def run():
        tools = {t.name for t in await server.list_tools()}
        result = await server.call_tool("search", {"query": "lentil soup", "top_k": 1})
        status = await server.call_tool("index_status", {})
        return tools, result, status

    tools, result, status = asyncio.run(run())
    assert tools == {"search", "read_indexed_file", "index_status"}
    assert "recipes.txt" in str(result)
    assert "'files_indexed': 3" in str(status) or '"files_indexed": 3' in str(status)
