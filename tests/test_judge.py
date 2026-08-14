"""Tests for the judge harness.

Two properties matter more than the rest. Judges must never be pooled — a mean
over two instruments looks exactly like a mean over one — and an unparseable
reply must not become a number, because a malformed reply scored as a default
is indistinguishable from a genuine middling verdict and drags every gap toward
zero.
"""

from __future__ import annotations

import json

import config
from src.eval import judge, swap_test


def record(**overrides) -> dict:
    base = {"judge": "gpt4o_mini", "judge_model": "gpt-4o-mini", "arm": "teacher",
            "poem_id": 1, "shown_id": 1, "condition": "matched",
            "score": 8, "reply": "8"}
    return {**base, **overrides}


def scored(poem_id: int, matched: int, random_: int, same_author: int,
           name: str = "gpt4o_mini") -> list[dict]:
    return [
        record(judge=name, poem_id=poem_id, condition="matched", score=matched),
        record(judge=name, poem_id=poem_id, condition="mismatched_random",
               score=random_),
        record(judge=name, poem_id=poem_id, condition="mismatched_same_author",
               score=same_author),
    ]


# --- parsing ----------------------------------------------------------------

def test_bare_integer_parses():
    assert judge.parse_score("7") == 7


def test_score_with_preamble_parses():
    assert judge.parse_score("Score: 9") == 9


def test_out_of_range_is_rejected():
    """A judge answering 50 on a 1-10 scale has not understood the task, and
    treating it as a score would silently distort the mean."""
    assert judge.parse_score("50") is None


def test_prose_reply_is_unscored_not_defaulted():
    """None, never a default. A malformed reply scored as 5 is indistinguishable
    from a genuine middling verdict and drags the gap toward zero."""
    assert judge.parse_score("This interpretation is quite good overall.") is None


def test_empty_reply_is_unscored():
    assert judge.parse_score("") is None


# --- judges are never pooled -------------------------------------------------

def test_single_judge_is_returned():
    assert judge.assert_single_judge(scored(1, 8, 3, 5)) == "gpt4o_mini"


def test_mixed_judges_raise_rather_than_average():
    """The failure mode that would quietly turn a robustness check into a
    composite metric nobody can interpret."""
    mixed = scored(1, 8, 3, 5) + scored(2, 7, 2, 4, name="gemini_flash")
    try:
        judge.assert_single_judge(mixed)
    except AssertionError:
        return
    raise AssertionError("a mixed-judge set was pooled instead of raising")


def test_aggregation_refuses_a_mixed_set():
    mixed = scored(1, 8, 3, 5) + scored(2, 7, 2, 4, name="gemini_flash")
    for call in (judge.condition_means, judge.gaps):
        try:
            call(mixed)
        except AssertionError:
            continue
        raise AssertionError(f"{call.__name__} pooled two judges")


def test_cache_paths_differ_by_judge():
    """Keyed by name in the FILENAME, so two judges cannot share a file even by
    mistake."""
    assert (judge.scores_path(config.PRIMARY_JUDGE)
            != judge.scores_path(config.SECONDARY_JUDGE))


# --- aggregation -------------------------------------------------------------

def test_condition_means():
    means = judge.condition_means(scored(1, 8, 2, 5) + scored(2, 6, 4, 3))
    assert means == {"matched": 7.0, "mismatched_random": 3.0,
                     "mismatched_same_author": 4.0}


def test_gaps_decompose_into_an_author_component():
    gaps = judge.gaps(scored(1, 8, 2, 5) + scored(2, 6, 4, 3))
    assert gaps["grounding_gap"] == 4.0
    assert gaps["poem_level_gap"] == 3.0
    assert gaps["author_component"] == 1.0
    assert gaps["n_pairs"] == 2


def test_author_component_is_zero_when_the_author_carries_nothing():
    """If the strict and standard controls score alike, none of the apparent
    grounding is author recognition."""
    gaps = judge.gaps(scored(1, 8, 3, 3) + scored(2, 9, 4, 4))
    assert gaps["author_component"] == 0.0


def test_unscored_pairs_are_excluded_not_counted_as_zero():
    records = scored(1, 8, 2, 5)
    records.append(record(poem_id=2, condition="matched", score=None))
    assert judge.condition_means(records)["matched"] == 8.0


def test_pairing_skips_poems_missing_a_condition():
    """A poem scored in only one condition contributes no difference — counting
    it as zero would bias the gap toward whichever condition survived."""
    partial = [record(poem_id=9, condition="matched", score=10)]
    assert judge.paired_differences(scored(1, 8, 2, 5) + partial,
                                    "matched", "mismatched_random") == [6]


# --- resumability ------------------------------------------------------------

def test_cache_returns_newest_record_per_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    path = judge.scores_path(config.PRIMARY_JUDGE)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(record(score=3)) + "\n")
        handle.write(json.dumps(record(score=9)) + "\n")

    cached = judge.load_cached(config.PRIMARY_JUDGE)
    assert len(cached) == 1 and cached[0]["score"] == 9


def test_pair_and_record_share_a_cache_key():
    """Otherwise nothing would ever be skipped on restart and every re-run would
    pay for the whole set again."""
    pair = swap_test.Pair(1, 1, "matched", "text", "teacher")
    assert judge._key(pair) == judge._key(record())


# --- retry policy -------------------------------------------------------------

class _Err(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def test_rate_limit_is_retried():
    assert judge._is_transient(_Err("rate limit", status_code=429))


def test_server_error_is_retried():
    assert judge._is_transient(_Err("bad gateway", status_code=502))


def test_spent_quota_is_not_retried():
    """A 429 that means "no credit" is permanent. Retrying it burns the backoff
    budget on every pair to reach the same answer, and buries the real cause."""
    assert not judge._is_transient(
        _Err("You exceeded your current quota, check billing", status_code=429))


def test_missing_model_is_not_retried():
    assert not judge._is_transient(_Err("model not found", status_code=404))


# --- summary and the gate -----------------------------------------------------

def test_summary_row_refuses_a_mixed_set():
    """A summary row that was secretly an average of two judges would be
    indistinguishable from one judge's result."""
    mixed = scored(1, 8, 3, 5) + scored(2, 7, 2, 4, name="gemini_flash")
    try:
        judge.summary_row(mixed)
    except AssertionError:
        return
    raise AssertionError("summary_row pooled two judges")


def test_summary_row_counts_unparseable_separately():
    records = scored(1, 8, 2, 5) + [record(poem_id=2, score=None)]
    row = judge.summary_row(records)
    assert (row["n_pairs"], row["n_scored"], row["n_unparseable"]) == (4, 3, 1)


def test_judges_occupy_separate_rows(tmp_path):
    table = judge.save_summary(
        [scored(1, 8, 2, 5), scored(1, 7, 3, 4, name="gemini_flash")],
        path=tmp_path / "summary.csv")
    assert len(table) == 2
    assert set(table["judge"]) == {"gpt4o_mini", "gemini_flash"}


def test_gate_passes_on_a_separating_judge():
    passed, message = judge.separation_verdict(scored(1, 9, 2, 3))
    assert passed and message.startswith("PASS")


def test_gate_fails_when_matched_and_mismatched_score_alike():
    """The case that invalidates the evaluation: the judge cannot tell grounded
    from ungrounded on text known to be grounded."""
    passed, message = judge.separation_verdict(scored(1, 6, 6, 6))
    assert not passed and "design must change" in message


def test_abort_counts_consecutive_failures_not_failures_since_start():
    """A daily quota exhausted midway leaves successes behind it. Counting
    failures-since-the-start never fires then, and the run grinds through every
    remaining pair to report the same error each time."""
    import inspect
    source = inspect.getsource(judge.score_all)
    assert "consecutive" in source
    assert "failures == index" not in source


# --- saturation and agreement -------------------------------------------------

def test_saturated_condition_is_flagged():
    """A control where every pair scored the same value makes the author
    component structurally zero — a property of the scale, not the model."""
    flat = [record(poem_id=i, condition="mismatched_random", score=1)
            for i in range(5)]
    assert "mismatched_random" in judge.saturated_conditions(flat)


def test_varied_condition_is_not_flagged():
    varied = [record(poem_id=i, condition="mismatched_random", score=s)
              for i, s in enumerate([1, 1, 2, 3])]
    assert judge.saturated_conditions(varied) == []


def test_spread_reports_range_and_counts():
    recs = [record(poem_id=i, condition="matched", score=s)
            for i, s in enumerate([8, 9, 9, 10])]
    stats = judge.score_spread(recs)["matched"]
    assert (stats["min"], stats["max"], stats["distinct"]) == (8, 10, 3)
    assert stats["counts"] == {8: 1, 9: 2, 10: 1}


def test_agreement_needs_both_judges_separately():
    """The two judges meet here and are still not pooled: separate arguments,
    separate columns, agreement BETWEEN rather than an average OF."""
    a = scored(1, 9, 1, 1) + scored(2, 8, 2, 1)
    b = scored(1, 10, 1, 1, name="gemini_flash") + \
        scored(2, 9, 1, 1, name="gemini_flash")
    result = judge.inter_judge_agreement(a, b)
    assert result["judges"] == ("gpt4o_mini", "gemini_flash")
    assert result["n"] == 6
    assert result["same_side"] == 1.0


def test_kappa_is_undefined_when_neither_judge_varies():
    """Perfect raw agreement on a constant is not evidence of agreement. With no
    variance there is no chance agreement to correct for, so kappa is undefined
    — and the NaN is the finding: both judges agree completely on a question
    neither could answer in more than one way."""
    import math

    a = [record(poem_id=i, condition="mismatched_random", score=1)
         for i in range(20)]
    b = [record(poem_id=i, condition="mismatched_random", score=1,
                judge="gemini_flash") for i in range(20)]
    result = judge.inter_judge_agreement(a, b)
    assert result["same_side"] == 1.0
    assert math.isnan(result["cohens_kappa"])


def test_kappa_is_defined_when_judges_vary():
    a = [record(poem_id=i, condition="matched", score=s)
         for i, s in enumerate([9, 9, 1, 1, 9, 1])]
    b = [record(poem_id=i, condition="matched", score=s, judge="gemini_flash")
         for i, s in enumerate([9, 1, 1, 1, 9, 9])]
    result = judge.inter_judge_agreement(a, b)
    assert 0.0 < result["cohens_kappa"] < 1.0


# --- reasoning is part of the instrument --------------------------------------

def test_reasoning_setting_keys_the_cache():
    """The same model deliberating and not deliberating must occupy two cache
    entries, or rescoring silently overwrites the comparison it exists for."""
    pair = swap_test.Pair(1, 1, "matched", "text", "teacher")
    assert judge._key(pair, "default") != judge._key(pair, "none")


def test_untagged_records_key_as_default():
    """Records written before the setting existed were the provider default,
    which is what they must continue to key as."""
    legacy = record()
    legacy.pop("reasoning", None)
    pair = swap_test.Pair(1, 1, "matched", "text", "teacher")
    assert judge._key(legacy) == judge._key(pair, "default")


def test_aggregation_refuses_to_mix_reasoning_settings():
    """A mean over a deliberating and a non-deliberating judge is a composite of
    two instruments wearing one name."""
    thinking = [record(poem_id=1, reasoning="default")]
    quiet = [record(poem_id=2, reasoning="none")]
    try:
        judge.assert_single_judge(thinking + quiet)
    except AssertionError:
        return
    raise AssertionError("two reasoning settings were pooled")


def test_same_setting_aggregates_normally():
    same = [record(poem_id=1, reasoning="none"),
            record(poem_id=2, reasoning="none")]
    assert judge.assert_single_judge(same) == "gpt4o_mini"


def test_primary_judge_sends_no_reasoning_parameter():
    """GPT-4o-mini is not a reasoning model and errors if the parameter is
    sent, so it must stay unset rather than be given a neutral value."""
    assert config.PRIMARY_JUDGE.reasoning is None


def test_secondary_judge_has_reasoning_disabled():
    assert config.SECONDARY_JUDGE.reasoning == "none"
