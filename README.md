# attic

A local, always-current semantic index of the files on your computer, searchable from
Claude through MCP. Notes, documents, PDFs, slides, spreadsheets and code go in; Claude
can then find "that internship report draft" or "where I wrote the router logic" without
you remembering the path.

Built on [kuhaku](https://github.com/ism00efe/kuhaku): kuhaku supplies the embeddings,
chunking, vector store and hybrid (vector + BM25) retrieval; attic adds the parts a
whole-computer index needs: scanning, exclude rules, incremental updates, crash safety,
a file watcher and an MCP server.

Everything stays on your machine. The embedding model runs locally (GPU if available).

## How it stays current

- **Incremental.** A SQLite manifest records each file's size, modification time and
  content hash. A sync only re-reads files whose size or time changed, and only
  re-embeds those whose *content* changed. A sync with nothing to do never even loads
  the model.
- **Deletes and renames** are picked up: chunks of files that disappeared are removed.
- **Safe to interrupt.** Kill it, close the laptop, lose power: the next sync resumes
  where it stopped. New versions are written before old ones are removed, and the
  manifest only moves forward after the store write succeeded, so a file is never
  missing from search because of a crash, and leftovers are swept at the start of every
  sync. (The tests simulate crashes at each step.)
- **One writer at a time**, enforced by an OS file lock that cannot go stale.
- **Watch mode** syncs once at start, then again whenever the folders have changed and
  been quiet for a while (5 minutes by default).

## What gets indexed

Text and notes (`.md .txt .rst .tex .org`), data (`.csv .json .yaml .toml .xml ...`),
web pages, code (Python, JS/TS, Java, C/C++, C#, Go, Rust and more), Jupyter notebooks,
PDF, Word (`.docx`), PowerPoint (`.pptx`) and Excel (`.xlsx`).

Skipped automatically:

- dependency trees, build output and caches: `node_modules`, virtualenvs (any folder
  with `pyvenv.cfg`), `__pycache__`, `build`, `dist`, `target`, `.git`, `AppData`, ...
- anything matched by a `.gitignore` in your projects
- hidden folders (configurable), empty files, files over 20 MB (configurable)
- **credentials, always**: `.ssh`, `.aws`, `.gnupg`, `.env*`, `*.pem`, `*.key`,
  `id_rsa*`, `*secret*`, `credentials*.json`, password databases, ...
- media and archives (not a supported type)

Card numbers, tokens, e-mail addresses, IP addresses, phone numbers and Turkish ID
numbers are masked before anything is embedded or stored (kuhaku's sanitizer).

Not covered yet: scanned PDFs (no OCR) and images.

## Install (Windows)

Python 3.11+.

1. Install PyTorch **with CUDA** first, so embedding runs on the GPU: pick the
   `pip install torch ...` command for your system on
   [pytorch.org](https://pytorch.org/get-started/locally/). (A plain `pip install`
   gives the CPU-only build on Windows. It works, just slower and hotter.)
2. Install attic:

   ```bash
   pip install git+https://github.com/ism00efe/attic
   ```

The first run downloads the embedding model (`intfloat/multilingual-e5-small`, about
0.5 GB, multilingual including Turkish).

## Use

```bash
attic init                 # writes ~/.attic/config.toml; list your folders under `roots`
attic sync --dry-run       # how many files would be indexed; embeds nothing
attic sync                 # first full index (the only heavy run), then incremental
attic search "internship report"
attic status               # files, chunks, last sync; --failed lists unparseable files
attic watch                # keep syncing as files change
```

The first `attic sync` is the only long, GPU-heavy run. To keep it cooler, set
`pause_between_batches_seconds` in the config (e.g. `2`): the work then runs in bursts
instead of flat out. Stop it any time; running it again continues.

## Connect to Claude

```bash
attic mcp-config
```

prints the entry to add under `mcpServers` in Claude Desktop's
`claude_desktop_config.json`. After restarting Claude Desktop, Claude gets three tools:

- `search(query, top_k, path_contains)`: passages with their file paths
- `read_indexed_file(path)`: the indexed text of a file a search found
- `index_status()`

The server is read-only and only returns what is in the index; it has no tool that
reads arbitrary paths, so excluded files stay out of reach.

To keep the index fresh while Claude Desktop is open, change `serve` to
`serve --watch` in that entry. Without it, run `attic watch` yourself or schedule
`attic sync` (e.g. Windows Task Scheduler); a running server picks up another
process's changes on its next search.

## Configuration

All keys except `roots` are optional; `attic init` writes a commented example.

| Key | Default | |
|---|---|---|
| `roots` | | folders to index |
| `exclude` | `[]` | extra gitignore-style patterns |
| `skip_hidden` | `true` | skip dot-folders (credentials stay excluded either way) |
| `respect_gitignore` | `true` | |
| `max_file_mb` | `20` | |
| `embedding_model` | `intfloat/multilingual-e5-small` | changing it needs `attic sync --rebuild` |
| `embedding_device` | `auto` | `auto`, `cuda`, `cpu` |
| `chunk_size` / `chunk_overlap` | `1000` / `150` | characters |
| `batch_files` | `32` | files embedded per batch |
| `pause_between_batches_seconds` | `0` | heat control for big runs |
| `low_priority` | `true` | run syncs at below-normal CPU priority |
| `watch_debounce_seconds` | `300` | quiet time before a watch-triggered sync |
| `retrieval` | `hybrid` | `hybrid` (vector + keyword) or `dense` |
| `data_dir` | `~/.attic` | index, manifest, lock and log |

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests run offline: a hashing fake replaces the embedding model and Chroma runs in a
temp directory, so they need no GPU and no model download.
