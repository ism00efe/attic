import os

import pytest

from attic.indexer import ModelChanged
from attic.lock import WriterBusy


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _bump_mtime(path):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


def _paths_in_store(idx):
    return {c.source_path for c in idx.store.iter_chunks()}


def test_first_sync_indexes_everything_and_second_embeds_nothing(root, make_indexer, fake):
    _write(root, "a.md", "kuhaku release notes")
    _write(root, "b.py", "def main():\n    return 1")
    _write(root, "c.txt", "groceries list")
    idx = make_indexer()
    r1 = idx.sync()
    assert (r1.indexed, r1.failed) == (3, 0)
    calls = fake.document_calls

    r2 = make_indexer().sync()
    assert r2.indexed == 0 and r2.unchanged == 3
    assert fake.document_calls == calls  # nothing re-embedded


def test_noop_sync_never_loads_the_model(root, cfg):
    from attic.indexer import Indexer

    _write(root, "a.md", "x")
    loads = []

    def factory(_cfg):
        loads.append(1)
        from tests.conftest import FakeEmbeddings

        return FakeEmbeddings()

    Indexer(cfg, embedder_factory=factory).sync()
    assert len(loads) == 1
    Indexer(cfg, embedder_factory=factory).sync()
    assert len(loads) == 1  # second sync found nothing to embed


def test_modified_file_replaces_its_old_chunks(root, make_indexer):
    p = _write(root, "plan.md", "old content about apples")
    idx = make_indexer()
    idx.sync()
    old_doc = idx.manifest.rows()[str(p)].doc_id

    p.write_text("new content about oranges", encoding="utf-8")
    _bump_mtime(p)
    r = idx.sync()
    assert r.indexed == 1
    texts = [c.text for c in idx.store.iter_chunks()]
    assert texts == ["new content about oranges"]
    assert old_doc not in idx.store.document_ids()


def test_deleted_file_disappears_from_index(root, make_indexer):
    keep = _write(root, "keep.md", "keep me")
    gone = _write(root, "gone.md", "delete me")
    idx = make_indexer()
    idx.sync()
    gone.unlink()
    r = idx.sync()
    assert r.removed == 1
    assert _paths_in_store(idx) == {str(keep)}
    assert str(gone) not in idx.manifest.rows()


def test_touched_but_identical_file_is_not_re_embedded(root, make_indexer, fake):
    p = _write(root, "a.md", "same bytes")
    idx = make_indexer()
    idx.sync()
    before = fake.texts_embedded
    _bump_mtime(p)
    r = idx.sync()
    assert r.content_unchanged == 1 and r.indexed == 0
    assert fake.texts_embedded == before
    # and the next sync sees it as plain unchanged (manifest mtime was updated)
    assert idx.sync().unchanged == 1


def test_unparseable_file_is_recorded_once_and_not_retried_until_it_changes(root, make_indexer):
    bad = root / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 garbage")
    idx = make_indexer()
    assert idx.sync().failed == 1
    assert idx.sync().failed == 0  # unchanged: not retried
    assert idx.manifest.stats()["failed"] == 1


def test_dry_run_reports_without_writing(root, make_indexer, fake):
    _write(root, "a.md", "x")
    idx = make_indexer()
    r = idx.sync(dry_run=True)
    assert r.plan_new == 1
    assert fake.document_calls == 0 and idx.store.count() == 0
    assert idx.manifest.rows() == {}


def test_moved_out_of_scope_root_is_removed(root, tmp_path, make_indexer):
    _write(root, "a.md", "x")
    idx = make_indexer()
    idx.sync()
    other = tmp_path / "other"
    other.mkdir()
    idx2 = make_indexer(roots=[str(other)])
    assert idx2.sync().removed == 1
    assert idx2.store.count() == 0


# ------------------------------------------------------------ interruption safety


def test_crash_mid_sync_then_resume_completes_without_duplicates(root, make_indexer, fake):
    for i in range(6):
        _write(root, f"f{i}.md", f"document number {i}")
    idx = make_indexer()  # batch_files=2 -> 3 batches
    fake.fail_after = 3  # first batch (2 texts) succeeds, second batch crashes
    with pytest.raises(RuntimeError, match="simulated crash"):
        idx.sync()
    assert len(idx.manifest.rows()) == 2  # only the committed batch is recorded

    fake.fail_after = None
    r = make_indexer().sync()
    assert r.indexed == 4 and r.unchanged == 2  # resumes, does not start over
    assert idx.store.count() == 6  # every file once, no duplicates


def test_crash_after_store_write_before_manifest_leaves_orphans_that_get_swept(
    root, make_indexer, monkeypatch
):
    _write(root, "a.md", "alpha")
    idx = make_indexer()

    def boom(rows):
        raise KeyboardInterrupt  # killed right after the chunks were written

    monkeypatch.setattr(idx.manifest, "upsert", boom)
    with pytest.raises(KeyboardInterrupt):
        idx.sync()
    assert idx.store.count() == 1 and idx.manifest.rows() == {}
    monkeypatch.undo()

    r = make_indexer().sync()
    assert r.orphans_removed == 1 and r.indexed == 1
    assert idx.store.count() == 1


def test_crash_after_manifest_before_old_version_deleted(root, make_indexer, monkeypatch):
    p = _write(root, "a.md", "version one")
    idx = make_indexer()
    idx.sync()
    p.write_text("version two", encoding="utf-8")
    _bump_mtime(p)

    real_delete = idx.store.delete_documents

    def boom(ids):
        raise KeyboardInterrupt

    monkeypatch.setattr(idx.store, "delete_documents", boom)
    with pytest.raises(KeyboardInterrupt):
        idx.sync()
    # both versions present for a moment; the file was never missing from search
    assert idx.store.count() == 2
    monkeypatch.setattr(idx.store, "delete_documents", real_delete)

    r = idx.sync()
    assert r.orphans_removed == 1
    assert [c.text for c in idx.store.iter_chunks()] == ["version two"]


def test_store_wiped_behind_the_manifest_is_rebuilt(root, make_indexer):
    _write(root, "a.md", "alpha")
    idx = make_indexer()
    idx.sync()
    idx.store.reset()
    r = make_indexer().sync()
    assert r.indexed == 1
    assert idx.store.count() == 1


def test_only_one_writer_at_a_time(root, make_indexer):
    _write(root, "a.md", "x")
    first = make_indexer()
    first.lock.acquire()
    try:
        with pytest.raises(WriterBusy):
            make_indexer().sync()
    finally:
        first.lock.release()


def test_changing_model_requires_rebuild(root, make_indexer):
    _write(root, "a.md", "x")
    make_indexer().sync()
    idx = make_indexer(embedding_model="another/model")
    with pytest.raises(ModelChanged):
        idx.sync()
    r = idx.sync(rebuild=True)
    assert r.indexed == 1 and idx.store.count() == 1


def test_sensitive_values_are_masked_before_embedding(root, make_indexer):
    token = "ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789"
    _write(root, "contacts.txt", f"Card 4111 1111 1111 1111 and token {token}")
    idx = make_indexer()
    idx.sync()
    stored = " ".join(c.text for c in idx.store.iter_chunks())
    assert "4111 1111" not in stored and "ghp_abcdef" not in stored
