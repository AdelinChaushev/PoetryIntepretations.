"""Tests for the prior-work comparison.

The failure this guards is a table that manufactures agreement. A missing
measurement, a qualitative reference or an absent interval must all yield
``not comparable`` — never a silent default to ``consistent``, which would put
a comparison in the report that was never made.

Direction must come from the data. Nothing here can prefer agreement.
"""

from __future__ import annotations

import config
from src.eval import prior_work as pw


# --- verdicts -----------------------------------------------------------------

def test_a_published_value_inside_our_interval_is_consistent():
    ref = pw.REFERENCES["judge_position_consistency"]      # 0.650
    assert pw.verdict(0.66, (0.60, 0.72), ref) == pw.CONSISTENT


def test_a_published_value_outside_our_interval_is_inconsistent():
    """The more interesting outcome, and it must be reachable."""
    ref = pw.REFERENCES["judge_position_consistency"]
    assert pw.verdict(0.40, (0.32, 0.48), ref) == pw.INCONSISTENT


def test_a_missing_measurement_is_not_comparable():
    ref = pw.REFERENCES["judge_position_consistency"]
    assert pw.verdict(None, (0.1, 0.9), ref) == pw.NOT_COMPARABLE
    assert pw.verdict(float("nan"), (0.1, 0.9), ref) == pw.NOT_COMPARABLE


def test_a_missing_interval_is_not_comparable():
    """A point estimate cannot be compared against a published value without
    saying how uncertain it is."""
    ref = pw.REFERENCES["judge_position_consistency"]
    assert pw.verdict(0.66, None, ref) == pw.NOT_COMPARABLE
    assert pw.verdict(0.66, (float("nan"), float("nan")), ref) == pw.NOT_COMPARABLE


def test_a_qualitative_reference_is_never_scored():
    """Gudibande et al. report a style/substance split, not a percentage.
    Scoring it numerically would invent a comparison."""
    ref = pw.REFERENCES["imitation_style_not_substance"]
    assert ref.value is None
    assert pw.verdict(0.5, (0.1, 0.9), ref) == pw.NOT_COMPARABLE


def test_nothing_defaults_to_agreement():
    """The failure the whole module exists to prevent."""
    ref = pw.REFERENCES["lima_sufficient_examples"]
    for measured, ci in ((None, None), (float("nan"), (0.0, 1.0)),
                         (500.0, None)):
        assert pw.verdict(measured, ci, ref) != pw.CONSISTENT


def test_interval_bounds_may_arrive_in_either_order():
    ref = pw.REFERENCES["judge_position_consistency"]
    assert pw.verdict(0.66, (0.72, 0.60), ref) == pw.CONSISTENT


# --- the references themselves ------------------------------------------------

def test_every_reference_records_its_setting():
    """The field that decides whether a comparison is like-for-like. Without it
    a table silently implies 0.5B and 65B are the same experiment."""
    for key, ref in pw.REFERENCES.items():
        assert ref.setting and len(ref.setting) > 20, key
        assert ref.citation and "arXiv" in ref.citation, key
        assert ref.locator, key


def test_every_reference_names_where_to_verify_it():
    """No number reaches the writeup from memory; the locator says where to
    check."""
    for key, ref in pw.REFERENCES.items():
        assert any(token in ref.locator
                   for token in ("§", "Table", "abstract", "Figure")), key


def test_the_judge_human_agreement_reference_is_flagged_as_weaker():
    """No human annotation exists here, so this one is a reference point rather
    than a warrant — and the table must say so."""
    ref = pw.REFERENCES["judge_human_agreement"]
    assert "not like-for-like" in ref.caveat.lower()


# --- the table ----------------------------------------------------------------

def test_a_citation_nobody_recorded_raises():
    """A comparison against a value that exists nowhere is not a comparison."""
    try:
        pw.compare([pw.Measurement("x", "no_such_reference")])
    except AssertionError as error:
        assert "not in REFERENCES" in str(error)
        return
    raise AssertionError("a table row cited a reference that does not exist")


def test_the_table_carries_the_setting_and_the_caveat():
    frame = pw.compare([pw.Measurement("flip rate", "judge_position_consistency",
                                       value=0.7, ci=(0.6, 0.8))])
    for column in ("published_setting", "citation", "locator", "caveat",
                   "verdict", "our_ci_low", "our_ci_high"):
        assert column in frame.columns
    assert frame.iloc[0]["published_setting"]


def test_both_judges_appear_as_separate_rows():
    """Never pooled: where they disagree, that disagreement is the finding."""
    frame = pw.compare([
        pw.Measurement("flip rate", "judge_position_consistency",
                       value=0.71, ci=(0.63, 0.78), judge="gpt4o_mini"),
        pw.Measurement("flip rate", "judge_position_consistency",
                       value=0.40, ci=(0.32, 0.48), judge="gemini_flash"),
    ])
    assert len(frame) == 2
    assert set(frame["judge"]) == {"gpt4o_mini", "gemini_flash"}
    assert set(frame["verdict"]) == {pw.CONSISTENT, pw.INCONSISTENT}


def test_the_table_is_written_not_typed(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PRIOR_WORK_CSV_PATH", tmp_path / "pw.csv")
    pw.build_comparison_table([
        pw.Measurement("flip rate", "judge_position_consistency",
                       value=0.7, ci=(0.6, 0.8))])
    assert (tmp_path / "pw.csv").exists()


def test_citations_used_lists_what_the_table_rests_on():
    """Each must be load-bearing in the body text, not parked in a
    bibliography."""
    frame = pw.compare([
        pw.Measurement("a", "judge_position_consistency", 0.7, (0.6, 0.8)),
        pw.Measurement("b", "lima_sufficient_examples"),
    ])
    citations = pw.citations_used(frame)
    assert len(citations) == 2
    assert any("Zheng" in c for c in citations)
