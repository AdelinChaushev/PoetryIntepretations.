"""Tests for swap-test pair construction.

Every constraint here fails silently. Nothing raises if the strict control
draws a near-duplicate, or if the standard control draws the same author — the
run completes and reports a plausible number that means something else. These
tests exist because the bug would otherwise surface as a finding.
"""

from __future__ import annotations

import collections

import config
from src.eval import swap_test


def poem(pid: int, author: str, title: str = "t") -> dict:
    return {"poem_id": pid, "author": author, "title": title,
            "lines": [f"line of poem {pid}"], "linecount": 1}


def interpretation(pid: int) -> dict:
    return {"poem_id": pid, "interpretation": f"about poem {pid}"}


#: Three authors, three poems each — enough for every condition to be drawable.
CORPUS = [poem(pid, author)
          for author, base in (("A", 0), ("B", 10), ("C", 20))
          for pid in range(base + 1, base + 4)]


def build(**kwargs):
    return swap_test.build_pairs([interpretation(1), interpretation(11)],
                                 CORPUS, **kwargs)


# --- structure ---------------------------------------------------------------

def test_every_poem_gets_all_three_conditions():
    pairs = build()
    by_poem = collections.Counter(p.poem_id for p in pairs)
    assert set(by_poem.values()) == {len(config.SWAP_CONDITIONS)}
    swap_test.assert_all_conditions_present(pairs)


def test_matched_shows_the_poem_the_interpretation_was_written_for():
    matched = [p for p in build() if p.condition == "matched"]
    assert all(p.shown_id == p.poem_id for p in matched)


def test_mismatched_never_shows_the_poem_itself():
    """A control that shows the poem IS the matched condition, and the gap it
    contributes to would collapse toward zero with nothing raised."""
    swap_test.assert_never_scored_against_itself(build())


def test_random_control_always_draws_a_different_author():
    """Otherwise it sometimes coincides with the same-author condition and the
    standard and strict gaps converge for the wrong reason."""
    swap_test.assert_random_is_different_author(build(), CORPUS)


def test_same_author_control_draws_the_same_author_but_another_poem():
    swap_test.assert_same_author_is_a_different_poem(build(), CORPUS)


# --- near-duplicates ---------------------------------------------------------

def test_near_duplicate_is_never_drawn_as_the_strict_control():
    """PoetryDB publishes poems under several titles. Drawing one turns the
    strict condition into the matched condition."""
    duplicates = {1: {2}, 2: {1}}
    pairs = swap_test.build_pairs([interpretation(1)], CORPUS,
                                  near_duplicates=duplicates)
    strict = [p for p in pairs if p.condition == "mismatched_same_author"]
    assert strict and all(p.shown_id == 3 for p in strict)
    swap_test.assert_no_near_duplicate_controls(pairs, duplicates)


def test_assertion_catches_a_near_duplicate_control():
    """The assertion must fail on a bad pair, not merely pass on a good one."""
    bad = [swap_test.Pair(1, 2, "mismatched_same_author", "x", "teacher")]
    try:
        swap_test.assert_no_near_duplicate_controls(bad, {1: {2}})
    except AssertionError:
        return
    raise AssertionError("a near-duplicate control was not caught")


def test_poem_with_only_duplicate_siblings_is_skipped_entirely():
    """Skipped, not emitted with two conditions. A poem missing one condition
    would drop out of the paired comparison and bias the gap."""
    corpus = [poem(1, "A"), poem(2, "A")]
    pairs = swap_test.build_pairs([interpretation(1)], corpus,
                                  near_duplicates={1: {2}, 2: {1}})
    assert pairs == []


# --- determinism -------------------------------------------------------------

def test_pairs_are_deterministic():
    """Judge calls are paid for and runs are resumable, so two partial runs must
    agree on which controls were drawn or their scores cannot be pooled."""
    swap_test.assert_deterministic([interpretation(1), interpretation(11)],
                                   CORPUS)


def test_a_different_seed_draws_different_controls():
    """Determinism must come from the seed, not from there being one choice."""
    corpus = [poem(pid, author)
              for author, base in (("A", 0), ("B", 10), ("C", 20))
              for pid in range(base + 1, base + 9)]
    drawn = {
        tuple(p.shown_id for p in swap_test.build_pairs(
            [interpretation(1)], corpus, seed=seed))
        for seed in (0, 1, 2, 3, 4)
    }
    assert len(drawn) > 1


# --- the composite check -----------------------------------------------------

def test_check_all_passes_on_a_valid_set():
    pairs = build()
    swap_test.check_all(pairs, CORPUS, near_duplicates={})


def test_check_all_rejects_a_control_showing_its_own_poem():
    bad = [
        swap_test.Pair(1, 1, "matched", "x", "teacher"),
        swap_test.Pair(1, 1, "mismatched_random", "x", "teacher"),
        swap_test.Pair(1, 2, "mismatched_same_author", "x", "teacher"),
    ]
    try:
        swap_test.check_all(bad, CORPUS)
    except AssertionError:
        return
    raise AssertionError("a control showing its own poem was not caught")
