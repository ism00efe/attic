"""MCP server: lets Claude search the index.

Read-only by design: the tools return what is in the index and nothing else. There is no
tool that reads an arbitrary path from disk, so the server cannot be used to reach files
the exclude rules kept out (credentials, .env files, excluded folders).
"""

from __future__ import annotations

import logging
import threading

from mcp.server.mcpserver import MCPServer

from .config import Config
from .search import Searcher

logger = logging.getLogger("attic")

INSTRUCTIONS = (
    "attic is a semantic index of the user's own files (notes, documents, PDFs, slides, "
    "spreadsheets, code) on their computer. Use `search` when the user refers to "
    "something they wrote, saved or worked on, or asks where something is on their "
    "computer. Results carry the file path; cite it. Use `read_indexed_file` to read "
    "more of a file a search turned up."
)


def build_server(cfg: Config, searcher: Searcher | None = None) -> MCPServer:
    searcher = searcher or Searcher(cfg)
    server = MCPServer(name="attic", instructions=INSTRUCTIONS)

    @server.tool()
    def search(query: str, top_k: int = 8, path_contains: str | None = None) -> list[dict]:
        """Semantic + keyword search over the user's indexed files.

        query: what to look for, in natural language (any language).
        top_k: how many passages to return (1-50).
        path_contains: optional case-insensitive substring the file path must contain,
            e.g. a folder or project name, to narrow the search.
        """
        return [hit.as_dict() for hit in searcher.search(query, top_k, path_contains)]

    @server.tool()
    def read_indexed_file(path: str, max_chars: int = 20000) -> str:
        """The indexed text of one file, by the exact path a search result returned."""
        text = searcher.document_text(path, max_chars=max(1000, min(max_chars, 100000)))
        return text if text is not None else f"Not in the index: {path}"

    @server.tool()
    def index_status() -> dict:
        """How many files are indexed, when the index last synced, which folders it covers."""
        return searcher.status()

    return server


def serve(cfg: Config, *, watch: bool = False) -> None:
    searcher: Searcher
    stop = threading.Event()
    if watch:
        from .autostart import running_pid

        if running_pid(cfg.data_path) is not None:
            # The background watcher already keeps the index current; a second one
            # would only collide with it on the writer lock.
            logger.info("serve: background watcher already running; not starting another")
            watch = False
    if watch:
        from .indexer import Indexer
        from .watcher import Watcher

        indexer = Indexer(cfg)
        # Shares the store (same-process writes are visible, see search.py) but not the
        # manifest: each thread keeps its own SQLite connection.
        searcher = Searcher(cfg, store=indexer.store)
        thread = threading.Thread(
            target=Watcher(indexer, debounce_seconds=cfg.watch_debounce_seconds).run,
            args=(stop,), name="attic-watch", daemon=True,
        )
        thread.start()
    else:
        searcher = Searcher(cfg)
    try:
        build_server(cfg, searcher).run("stdio")
    finally:
        stop.set()
