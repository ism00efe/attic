from attic.ignore import Walker


def _names(root, **kw):
    return sorted(p.path.relative_to(root).as_posix() for p in Walker([root], **kw).walk())


def _touch(root, rel, text="hello"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_skips_dependency_trees_caches_and_unsupported_types(root):
    _touch(root, "notes/plan.md")
    _touch(root, "proj/main.py")
    _touch(root, "proj/node_modules/lib/index.js")
    _touch(root, "proj/__pycache__/x.py")
    _touch(root, "proj/build/out.txt")
    _touch(root, "video.mp4")
    _touch(root, "archive.zip")
    assert _names(root) == ["notes/plan.md", "proj/main.py"]


def test_virtualenv_under_any_name_is_skipped(root):
    _touch(root, "myenv/pyvenv.cfg", "home = x")
    _touch(root, "myenv/Lib/site.py")
    _touch(root, "real.py")
    assert _names(root) == ["real.py"]


def test_secrets_are_always_excluded_even_when_hidden_files_are_allowed(root):
    _touch(root, ".ssh/config.txt")
    _touch(root, ".env")
    _touch(root, "proj/.env.local")
    _touch(root, "keys/server.key")
    _touch(root, "client_secret.json")
    _touch(root, ".obsidian/notes.md")
    assert _names(root, skip_hidden=False) == [".obsidian/notes.md"]


def test_hidden_folders_skipped_by_default(root):
    _touch(root, ".config/app.json")
    _touch(root, "visible.txt")
    assert _names(root) == ["visible.txt"]


def test_gitignore_is_respected_per_directory(root):
    _touch(root, "repo/.gitignore", "data/\n*.csv\n")
    _touch(root, "repo/data/big.txt")
    _touch(root, "repo/table.csv")
    _touch(root, "repo/src/app.py")
    _touch(root, "other/table.csv")
    assert _names(root) == ["other/table.csv", "repo/src/app.py"]
    assert len(_names(root, respect_gitignore=False)) == 4


def test_user_excludes_and_size_limit(root):
    _touch(root, "Downloads/old/a.txt")
    _touch(root, "Downloads/new/b.txt")
    _touch(root, "big.txt", "x" * 2000)
    _touch(root, "empty.txt", "")
    assert _names(root, exclude=["Downloads/old/"], max_file_bytes=1000) == [
        "Downloads/new/b.txt"
    ]


def test_is_wanted_filters_watch_events(root):
    w = Walker([root])
    assert w.is_wanted(root / "notes" / "a.md")
    assert not w.is_wanted(root / "proj" / "node_modules" / "a.js")
    assert not w.is_wanted(root / "movie.mkv")
    assert not w.is_wanted(root.parent / "outside.md")
