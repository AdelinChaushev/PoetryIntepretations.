"""Tests for the provenance manifest.

A manifest that passes verification when the data has changed is worse than no
manifest: it converts an unchecked claim into a false certificate. These tests
mostly check that verification FAILS when it should.
"""

from __future__ import annotations

import json

import config
from src import provenance


def write(path, text: str = "one\ntwo\n"):
    path.write_text(text, encoding="utf-8")
    return path


# --- digests -----------------------------------------------------------------

def test_digest_records_content_size_and_lines(tmp_path):
    record = provenance.file_digest(write(tmp_path / "a.jsonl"))
    assert record["present"] and record["lines"] == 2
    assert len(record["sha256"]) == 64


def test_missing_file_is_marked_absent_not_skipped(tmp_path):
    """A manifest that quietly omitted an absent input would certify a run that
    never had it."""
    assert provenance.file_digest(tmp_path / "nope.jsonl") == {"present": False}


def test_identical_content_digests_identically(tmp_path):
    a = provenance.file_digest(write(tmp_path / "a"))["sha256"]
    b = provenance.file_digest(write(tmp_path / "b"))["sha256"]
    assert a == b


def test_one_changed_byte_changes_the_digest(tmp_path):
    before = provenance.file_digest(write(tmp_path / "a"))["sha256"]
    after = provenance.file_digest(write(tmp_path / "a", "one\ntwO\n"))["sha256"]
    assert before != after


# --- verification ------------------------------------------------------------

def test_fresh_manifest_verifies_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_POEMS_PATH", write(tmp_path / "raw.jsonl"))
    path = tmp_path / "manifest.json"
    provenance.save_manifest(path)
    assert provenance.verify(path) == []


def test_edited_artifact_is_detected(tmp_path, monkeypatch):
    """The case the manifest exists for: data changed after the numbers were
    reported, and nothing else would notice."""
    artifact = write(tmp_path / "raw.jsonl")
    monkeypatch.setattr(config, "RAW_POEMS_PATH", artifact)
    path = tmp_path / "manifest.json"
    provenance.save_manifest(path)

    write(artifact, "one\ntwo\nthree\n")
    problems = provenance.verify(path)
    assert len(problems) == 1 and "content changed" in problems[0]


def test_deleted_artifact_is_detected(tmp_path, monkeypatch):
    artifact = write(tmp_path / "raw.jsonl")
    monkeypatch.setattr(config, "RAW_POEMS_PATH", artifact)
    path = tmp_path / "manifest.json"
    provenance.save_manifest(path)

    artifact.unlink()
    assert any("MISSING" in p for p in provenance.verify(path))


def test_changed_setting_is_detected(tmp_path, monkeypatch):
    """A threshold changed after the fact would silently reinterpret every
    number the manifest describes."""
    path = tmp_path / "manifest.json"
    provenance.save_manifest(path)

    monkeypatch.setattr(config, "MIN_LINES", config.MIN_LINES + 1)
    assert any("MIN_LINES" in p for p in provenance.verify(path))


def test_missing_manifest_reports_rather_than_raises(tmp_path):
    assert provenance.verify(tmp_path / "absent.json") != []


def test_verify_reports_every_problem_not_just_the_first(tmp_path, monkeypatch):
    """A reader checking someone else's repository wants the full picture."""
    artifact = write(tmp_path / "raw.jsonl")
    monkeypatch.setattr(config, "RAW_POEMS_PATH", artifact)
    path = tmp_path / "manifest.json"
    provenance.save_manifest(path)

    write(artifact, "changed\n")
    monkeypatch.setattr(config, "SEED", config.SEED + 1)
    assert len(provenance.verify(path)) >= 2


# --- what the manifest carries -----------------------------------------------

def test_manifest_records_the_settings_that_change_results():
    settings = provenance.build_manifest()["settings"]
    for name in ("SEED", "MIN_LINES", "MAX_SEQ_LEN", "N_FOLDS",
                 "JUDGE_TEMPERATURE"):
        assert name in settings


def test_manifest_flags_a_dirty_tree():
    """A result from a dirty tree cannot be traced to any version of the code,
    which matters more than the commit id itself."""
    assert "dirty" in provenance.build_manifest()["git"]


def test_manifest_is_json_serialisable(tmp_path):
    path = tmp_path / "manifest.json"
    provenance.save_manifest(path)
    json.loads(path.read_text(encoding="utf-8"))


# --- determinism of the stages that have no API in them -----------------------

def test_the_deterministic_pipeline_is_deterministic():
    """Everything after generation is a pure function of the cached data. If
    this ever fails, a reported number cannot be re-derived from the artifacts
    the repo ships, and the manifest certifies nothing."""
    from src.data import fetch_poems, generate, splits
    from src.data import filter as data_filter

    poems = fetch_poems.load_cached()
    interpretations = generate.load_cached()
    if not poems or not interpretations:
        return  # nothing cached in this environment

    first, funnel_a = data_filter.build_corpus(poems, interpretations,
                                               check_tokens=False)
    second, funnel_b = data_filter.build_corpus(poems, interpretations,
                                                check_tokens=False)
    assert [p["poem_id"] for p in first] == [p["poem_id"] for p in second]
    assert funnel_a == funnel_b

    # SMOKE caps the corpus at N_POEMS, which can leave fewer authors than
    # folds — the partition then cannot exist, and make_folds says so rather
    # than inventing one. Corpus determinism above is still worth asserting.
    if len({p["author"] for p in first}) < config.N_FOLDS:
        return

    folds_a = splits.make_folds(first, config.N_FOLDS,
                                group_key=config.FOLD_GROUP_KEY,
                                seed=config.SEED)
    folds_b = splits.make_folds(second, config.N_FOLDS,
                                group_key=config.FOLD_GROUP_KEY,
                                seed=config.SEED)
    assert folds_a.sizes() == folds_b.sizes()
    assert [[p["poem_id"] for p in group] for group in folds_a.held_out] == \
           [[p["poem_id"] for p in group] for group in folds_b.held_out]


def test_poem_ids_are_derived_from_content_not_order():
    """Ids key the interpretations and the fold assignment. If they depended on
    fetch order, re-fetching would repoint every stored record at a different
    poem and nothing would raise."""
    from src.data import fetch_poems

    poems = fetch_poems.load_cached()
    if not poems:
        return
    shuffled = list(reversed(poems))
    renumbered = {p["poem_id"]: (p["author"], p["title"])
                  for p in fetch_poems.assign_ids(shuffled)}
    original = {p["poem_id"]: (p["author"], p["title"]) for p in poems}
    assert renumbered == original
