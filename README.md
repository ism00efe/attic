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
- **Unchanged text is never re-embedded.** A touched or re-saved file whose extracted
  text is the same keeps its vectors.
- **Runs in the background.** After the first `attic sync`, a watcher starts and is set
  to start at every login (Windows). It syncs whenever the folders have changed and been
  quiet for a while (5 minutes by default). `attic autostart off` turns it off.

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

Optionally, `mask_sensitive = true` masks e-mail addresses, phone numbers, card
numbers, tokens, IP addresses and Turkish ID numbers before anything is embedded or
stored (kuhaku's sanitizer). It is off by default: this is a private index of your own
files, and masking makes questions like "what was X's phone number" unanswerable.

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

The first run downloads the embedding model,
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (740M parameters,
Apache-2.0, 100+ languages including Turkish, 8K-token context; runs in bfloat16 on the
GPU).

## Use

```bash
attic init                 # writes ~/.attic/config.toml; list your folders under `roots`
attic sync --dry-run       # how many files would be indexed; embeds nothing
attic sync                 # first full index (the only heavy run), then incremental
attic search "internship report"
attic status               # files, chunks, last sync; --failed lists unparseable files
```

The first `attic sync` is the only long, GPU-heavy run. To keep it cooler, set
`pause_between_batches_seconds` in the config (e.g. `2`): the work then runs in bursts
instead of flat out. Stop it any time; running it again continues.

### Background watcher

When the first `attic sync` finishes, attic starts `attic watch` in the background
(no window) and registers it to start at every login, through the per-user Run key: no
admin rights, and it is listed under Task Manager > Startup apps. At each start it syncs
whatever changed while it was off, which is incremental and quick.

```bash
attic autostart status     # on/off, running or not
attic autostart off        # stop it and remove it from login; remembered
attic autostart on         # register again and start it now
attic stop                 # stop it for now; it starts again at next login
```

Autostart is implemented for Windows. On other systems, `attic watch` runs in the
foreground; add it to your login items yourself.

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

The background watcher keeps the index fresh; the server picks up its changes on the
next search. If you turned autostart off but want syncing while Claude Desktop is open,
use `serve --watch` in that entry instead.

## Configuration

All keys except `roots` are optional; `attic init` writes a commented example.

| Key | Default | |
|---|---|---|
| `roots` | | folders to index |
| `exclude` | `[]` | extra gitignore-style patterns |
| `skip_hidden` | `true` | skip dot-folders (credentials stay excluded either way) |
| `respect_gitignore` | `true` | |
| `max_file_mb` | `20` | |
| `mask_sensitive` | `false` | mask e-mails, phone/card/ID numbers, tokens, IPs |
| `embedding_model` | `google/embeddinggemma-2` | changing it needs `--rebuild` |
| `embedding_dims` | full (768) | Matryoshka truncation; changing it needs `--rebuild` |
| `embedding_device` | `auto` | `auto`, `cuda`, `cpu` |
| `embedding_batch_size` | `16` | chunks per GPU batch; lower it if memory runs out |
| `chunk_size` / `chunk_overlap` | `1000` / `150` | characters |
| `batch_files` | `32` | files embedded per batch |
| `pause_between_batches_seconds` | `0` | heat control for big runs |
| `low_priority` | `true` | run syncs at below-normal CPU priority |
| `watch_debounce_seconds` | `300` | quiet time before a watch-triggered sync |
| `retrieval` | `hybrid` | `hybrid` (vector + keyword) or `dense` |
| `data_dir` | `~/.attic` | index, manifest, lock and log |

Changing `chunk_size`, `chunk_overlap` or `mask_sensitive` later needs nothing special:
the next sync re-reads every file and re-embeds only the ones whose chunks come out
different (turning masking on, for example, only touches files that contain something
to mask). Changing `embedding_model` or `embedding_dims` is the exception: vectors from
two models or sizes cannot be mixed, so every file must be embedded again. attic stops
and asks you to run `attic sync --rebuild` when you are ready for that.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests run offline: a hashing fake replaces the embedding model and Chroma runs in a
temp directory, so they need no GPU and no model download.

## License

Apache-2.0, see [LICENSE](https://github.com/ism00efe/attic/blob/main/LICENSE).
