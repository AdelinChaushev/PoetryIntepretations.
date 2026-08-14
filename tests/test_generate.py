"""Tests for the interpretation cache and the teacher measurements taken from it.

The cache is append-only and resampling writes a second record for a poem, so
"the newest record wins" is right for the interpretation text and wrong for
every number reported *about* the teacher. A later success overwriting an
earlier failure would report a teacher that does not misquote — which is the
exact failure this project was built to detect, appearing in its own pipeline.
Both accumulated fields are pinned here because nothing raises when they break.
"""

from __future__ import annotations

import json

import config
from src.data import generate
from src.data import filter as data_filter


POEM = {"poem_id": 1, "title": "t", "author": "a",
        "lines": ["the cat sat on the mat"], "linecount": 1}


def write_cache(tmp_path, monkeypatch, records: list[dict]):
    """Point the cache at a temporary JSONL holding ``records``."""
    path = tmp_path / "interpretations.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    monkeypatch.setattr(config, "INTERPRETATIONS_PATH", path)
    return path


def record(**overrides) -> dict:
    base = {"poem_id": 1, "title": "t", "author": "a",
            "interpretation": 'It says "the cat sat on the mat" plainly.',
            "attempts": 1, "grounded": True, "first_attempt_grounded": True}
    return {**base, **overrides}


# --- what the newest record wins, and what it does not ----------------------

def test_newest_interpretation_wins(tmp_path, monkeypatch):
    write_cache(tmp_path, monkeypatch, [
        record(interpretation="first"), record(interpretation="second"),
    ])
    assert generate.load_cached()[0]["interpretation"] == "second"


def test_first_attempt_failure_survives_a_later_success(tmp_path, monkeypatch):
    """The measurement of the teacher must not improve when the poem is saved."""
    write_cache(tmp_path, monkeypatch, [
        record(grounded=False, first_attempt_grounded=False),
        record(grounded=True, first_attempt_grounded=True),
    ])
    assert generate.load_cached()[0]["first_attempt_grounded"] is False


def test_attempts_accumulate_across_runs(tmp_path, monkeypatch):
    """`attempts` counts calls within one run. A poem that failed in one run and
    succeeded on the first call of the next stores attempts=1 twice, having
    genuinely cost two calls."""
    write_cache(tmp_path, monkeypatch, [
        record(attempts=1, grounded=False), record(attempts=1, grounded=True),
    ])
    assert generate.load_cached()[0]["total_attempts"] == 2


def test_single_record_reports_its_own_attempts(tmp_path, monkeypatch):
    write_cache(tmp_path, monkeypatch, [record(attempts=2)])
    assert generate.load_cached()[0]["total_attempts"] == 2


# --- the reported hallucination rate ----------------------------------------

def test_rate_reads_first_attempts_not_stored_text():
    """The stored text is grounded; the first attempt was not. Recomputing from
    the text would report 0% — a teacher that never misquotes."""
    saved = record(grounded=True, first_attempt_grounded=False)
    rate, _ = data_filter.hallucination_rate([saved], [POEM])
    assert rate == 1.0


def test_rate_falls_back_to_the_text_when_untracked():
    """Records predating attempt tracking carry no flag, and for those the
    stored text *is* the first attempt."""
    untracked = {"poem_id": 1, "interpretation": 'It says "a line never written".'}
    rate, _ = data_filter.hallucination_rate([untracked], [POEM])
    assert rate == 1.0


def test_rate_ignores_interpretations_without_a_poem():
    orphan = record(poem_id=999, first_attempt_grounded=False)
    rate, _ = data_filter.hallucination_rate([record(), orphan], [POEM])
    assert rate == 0.0


# --- the attempt distribution ------------------------------------------------

def test_distribution_counts_accumulated_attempts():
    counted = data_filter.attempt_distribution([
        record(total_attempts=1), record(total_attempts=2),
        record(total_attempts=2), record(total_attempts=3),
    ])
    assert counted["by_attempts"] == {1: 1, 2: 2, 3: 1}
    assert counted["resampled"] == 3


def test_distribution_falls_back_to_the_flag_without_poems():
    counted = data_filter.attempt_distribution([
        record(total_attempts=3, grounded=False), record(total_attempts=1),
    ])
    assert counted["ungrounded"] == 1
    assert counted["ungrounded_rate"] == 0.5


def test_distribution_rechecks_against_poems_rather_than_the_flag():
    """The flag is written at generation time and a later checker fix cannot
    reach it. This one says ungrounded; the current checker disagrees."""
    stale = record(grounded=False)
    assert data_filter.attempt_distribution([stale])["ungrounded"] == 1
    assert data_filter.attempt_distribution([stale], [POEM])["ungrounded"] == 0


def test_stored_basis_and_first_attempt_basis_disagree_after_resampling():
    """The two bases must not be interchangeable. A record whose first attempt
    misquoted and whose stored text does not is the whole reason the flag is
    written at generation time — reading the stored text reports 0%."""
    saved = record(grounded=True, first_attempt_grounded=False)

    teacher, _ = data_filter.hallucination_rate([saved], [POEM])
    corpus, _ = data_filter.hallucination_rate([saved], [POEM],
                                               first_attempt_only=False)
    assert (teacher, corpus) == (1.0, 0.0)


def test_bases_agree_when_nothing_was_resampled():
    grounded = record(grounded=True, first_attempt_grounded=True)
    assert (data_filter.hallucination_rate([grounded], [POEM])[0]
            == data_filter.hallucination_rate([grounded], [POEM],
                                              first_attempt_only=False)[0])


def test_first_attempts_returns_the_oldest_record(tmp_path, monkeypatch):
    """The mirror of load_cached. Scoring these with the current checker is what
    makes the teacher rate reproducible after a checker fix."""
    write_cache(tmp_path, monkeypatch, [
        record(interpretation="original"), record(interpretation="resampled"),
    ])
    assert generate.first_attempts()[0]["interpretation"] == "original"
    assert generate.load_cached()[0]["interpretation"] == "resampled"


# --- retrying poems that never grounded --------------------------------------

def test_exhausted_poem_is_skipped_by_default(tmp_path, monkeypatch):
    write_cache(tmp_path, monkeypatch,
                [record(attempts=config.GENERATE_MAX_ATTEMPTS, grounded=False,
                        interpretation='It says "a line never written".')])
    assert generate.processed_ids([POEM]) == {1}


def test_retry_ungrounded_re_queues_it(tmp_path, monkeypatch):
    """`attempts` is per-run, so without an override a poem that exhausted one
    run is skipped by every future run — permanently, with no way back."""
    write_cache(tmp_path, monkeypatch,
                [record(attempts=config.GENERATE_MAX_ATTEMPTS, grounded=False,
                        interpretation='It says "a line never written".')])
    assert generate.processed_ids([POEM], retry_ungrounded=True) == set()


def test_retry_stops_at_the_total_ceiling(tmp_path, monkeypatch):
    """Otherwise re-running grants three more calls indefinitely."""
    write_cache(tmp_path, monkeypatch, [
        record(attempts=3, grounded=False,
               interpretation='It says "a line never written".')
    ] * config.GENERATE_MAX_TOTAL_ATTEMPTS)
    assert generate.processed_ids([POEM], retry_ungrounded=True) == {1}


def test_stale_flag_does_not_re_queue_a_now_grounded_poem(tmp_path, monkeypatch):
    """The 13 underscore-markup false negatives: flagged ungrounded by the old
    checker, correct under the current one. Re-running must not pay to redo them."""
    write_cache(tmp_path, monkeypatch, [record(attempts=3, grounded=False)])
    assert generate.processed_ids([POEM], retry_ungrounded=True) == {1}


# --- which evidence is authoritative for the teacher rate --------------------

def test_teacher_rate_rescores_single_attempt_text(tmp_path, monkeypatch):
    """attempts == 1: the stored text IS the first attempt, so a checker fix
    must reach it. The stale flag here says grounded; the text does not."""
    write_cache(tmp_path, monkeypatch, [
        record(attempts=1, first_attempt_grounded=True,
               interpretation='It says "a line never written".'),
    ])
    rate, _, n = data_filter.teacher_hallucination_rate([POEM])
    assert (rate, n) == (1.0, 1)


def test_teacher_rate_trusts_the_flag_when_text_is_a_later_attempt(tmp_path,
                                                                   monkeypatch):
    """attempts > 1: interpret_until_grounded writes only the last text, so the
    first attempt's wording is gone and the flag is the only evidence left.
    Re-scoring the survivor would erase a known failure."""
    write_cache(tmp_path, monkeypatch, [
        record(attempts=2, first_attempt_grounded=False),   # stored text grounds
    ])
    rate, _, n = data_filter.teacher_hallucination_rate([POEM])
    assert (rate, n) == (1.0, 1)


def test_teacher_rate_reads_the_oldest_record(tmp_path, monkeypatch):
    write_cache(tmp_path, monkeypatch, [
        record(attempts=1, interpretation='It says "a line never written".'),
        record(attempts=1, interpretation='It says "the cat sat on the mat".'),
    ])
    rate, _, n = data_filter.teacher_hallucination_rate([POEM])
    assert (rate, n) == (1.0, 1)


# --- saying why poems were held back -----------------------------------------

def test_warns_when_ungrounded_poems_are_not_retried(tmp_path, monkeypatch, caplog):
    """"nothing to generate" must not be the only thing said when poems were
    skipped — that reads as success to anyone who just asked for a retry."""
    write_cache(tmp_path, monkeypatch, [
        record(attempts=3, grounded=False,
               interpretation='It says "a line never written".'),
    ])
    with caplog.at_level("WARNING"):
        generate._report_withheld([POEM], {1}, retry_ungrounded=False)
    assert "retry_ungrounded=True" in caplog.text


def test_warns_when_the_ceiling_is_what_stopped_it(tmp_path, monkeypatch, caplog):
    # total_attempts is derived by summing across records, so it cannot be set
    # directly — the ceiling is reached by accumulating runs, as in real use.
    write_cache(tmp_path, monkeypatch, [
        record(attempts=3, grounded=False,
               interpretation='It says "a line never written".'),
    ] * (config.GENERATE_MAX_TOTAL_ATTEMPTS // 3))
    with caplog.at_level("WARNING"):
        generate._report_withheld([POEM], {1}, retry_ungrounded=True)
    assert "GENERATE_MAX_TOTAL_ATTEMPTS" in caplog.text


def test_silent_when_everything_grounded(tmp_path, monkeypatch, caplog):
    write_cache(tmp_path, monkeypatch, [record()])
    with caplog.at_level("WARNING"):
        generate._report_withheld([POEM], {1}, retry_ungrounded=True)
    assert caplog.text == ""
