"""Command line entry point: ``attic <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import EXAMPLE_CONFIG, Config, default_config_path, load_config


def _setup_logging(cfg: Config | None, verbose: bool) -> None:
    root = logging.getLogger("attic")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    # stderr only: in `serve`, stdout is the MCP channel and any stray print breaks it.
    err = logging.StreamHandler(sys.stderr)
    err.setFormatter(fmt)
    root.addHandler(err)
    if cfg is not None:
        cfg.data_path.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(cfg.data_path / "attic.log", maxBytes=2_000_000,
                                 backupCount=2, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def _cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.config).expanduser() if args.config else default_config_path()
    if path.exists() and not args.force:
        print(f"Config already exists: {path} (use --force to overwrite)")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(EXAMPLE_CONFIG, encoding="utf-8")
    print(f"Wrote {path}\nEdit `roots`, then run `attic sync --dry-run` to preview.")
    return 0


def _progress(done: int, total: int) -> None:
    print(f"\r  {done}/{total} files", end="", file=sys.stderr, flush=True)
    if done == total:
        print(file=sys.stderr)


def _cmd_sync(cfg: Config, args: argparse.Namespace) -> int:
    from .indexer import Indexer

    report = Indexer(cfg).sync(dry_run=args.dry_run, rebuild=args.rebuild, progress=_progress)
    print(report.summary())
    return 0


def _cmd_watch(cfg: Config, args: argparse.Namespace) -> int:
    from .indexer import Indexer
    from .watcher import Watcher

    stop = threading.Event()
    watcher = Watcher(Indexer(cfg), debounce_seconds=cfg.watch_debounce_seconds)
    thread = threading.Thread(target=watcher.run, args=(stop,), daemon=True)
    thread.start()
    print(f"Watching {len(cfg.root_paths)} folder(s). Ctrl+C to stop.", file=sys.stderr)
    try:
        while thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop.set()
        thread.join(timeout=30)
    return 0


def _cmd_serve(cfg: Config, args: argparse.Namespace) -> int:
    from .server import serve

    serve(cfg, watch=args.watch)
    return 0


def _cmd_search(cfg: Config, args: argparse.Namespace) -> int:
    from .search import Searcher

    hits = Searcher(cfg).search(args.query, args.k, args.path)
    if not hits:
        print("No results.")
    for hit in hits:
        snippet = " ".join(hit.text.split())[:240]
        print(f"{hit.score:.3f}  {hit.path}\n       {snippet}\n")
    return 0


def _cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    from .search import Searcher

    searcher = Searcher(cfg)
    status = searcher.status()
    if status["last_sync_at"]:
        status["last_sync_at"] = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(status["last_sync_at"])
        )
    print(json.dumps(status, indent=2, ensure_ascii=False))
    if args.failed:
        for path, error in searcher.manifest.failed():
            print(f"FAILED {path}: {error}")
    return 0


def _cmd_mcp_config(cfg_path: Path) -> int:
    exe = shutil.which("attic") or sys.argv[0]
    snippet = {
        "mcpServers": {
            "attic": {
                "command": str(Path(exe).resolve()),
                "args": ["--config", str(cfg_path.resolve()), "serve"],
            }
        }
    }
    print(json.dumps(snippet, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="attic", description=__doc__)
    parser.add_argument("--config", help="path to config.toml (default ~/.attic/config.toml)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="write an example config")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("sync", help="index new and changed files, drop deleted ones")
    p.add_argument("--dry-run", action="store_true",
                   help="only report what would change; embeds nothing")
    p.add_argument("--rebuild", action="store_true",
                   help="drop the index and re-embed everything (after changing the model)")

    sub.add_parser("watch", help="sync now, then keep syncing as files change")

    p = sub.add_parser("serve", help="run the MCP server (stdio) for Claude")
    p.add_argument("--watch", action="store_true", help="also keep the index synced")

    p = sub.add_parser("search", help="search from the terminal")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--path", help="only paths containing this text")

    p = sub.add_parser("status", help="index size and last sync")
    p.add_argument("--failed", action="store_true", help="list files that failed to parse")

    sub.add_parser("mcp-config", help="print the Claude Desktop config entry for attic")

    args = parser.parse_args(argv)
    if args.command == "init":
        return _cmd_init(args)
    cfg_path = Path(args.config).expanduser() if args.config else default_config_path()
    if args.command == "mcp-config":
        return _cmd_mcp_config(cfg_path)

    try:
        cfg = load_config(cfg_path)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    _setup_logging(cfg, args.verbose)
    if not cfg.roots and args.command in ("sync", "watch"):
        print("No `roots` in the config: nothing to index.", file=sys.stderr)
        return 2

    from .indexer import ModelChanged
    from .lock import WriterBusy

    handlers = {"sync": _cmd_sync, "watch": _cmd_watch, "serve": _cmd_serve,
                "search": _cmd_search, "status": _cmd_status}
    try:
        return handlers[args.command](cfg, args)
    except (WriterBusy, ModelChanged) as exc:
        print(exc, file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
