"""Tests for the memorisation probe.

The probe exists to turn a confound into a measurement: public-domain poetry is
in pretraining data, so the base model may recite rather than read. What matters
is not the headline share but the per-poem flag, because every result is
stratified by it — a grounding gap that survives on non-memorised poems is not
recall, and one that collapses there is.
"""

from __future__ import annotations

import config
from src.eval import contamination


POEM = {"poem_id": 1, "author": "A", "title": "T", "linecount": 5,
        "lines": ["Some say the world will end in fire",
                  "Some say in ice",
                  "From what I have tasted of desire",
                  "I hold with those who favour fire",
                  "But if it had to perish twice"]}


# --- the prompt ---------------------------------------------------------------

def test_prompt_uses_only_the_opening_lines():
    prompt = contamination.continuation_prompt(POEM, n_lines=2)
    assert prompt == "Some say the world will end in fire\nSome say in ice"


def test_prompt_carries_no_title_or_author():
    """Naming the poem would let a model that recognises the TITLE recite from
    that alone — a different and weaker claim than recognising the text."""
    prompt = contamination.continuation_prompt(POEM)
    assert POEM["title"] not in prompt and POEM["author"] not in prompt


# --- scoring ------------------------------------------------------------------

def test_exact_recitation_scores_one():
    rest = "\n".join(POEM["lines"][config.CONTAMINATION_PROMPT_LINES:])
    assert contamination.reproduction_rate(rest, POEM) == 1.0


def test_unrelated_continuation_scores_zero():
    assert contamination.reproduction_rate(
        "A completely different poem about the sea and its tides", POEM) == 0.0


def test_partial_recall_scores_in_between():
    partial = POEM["lines"][2]
    rate = contamination.reproduction_rate(partial, POEM)
    assert 0 < rate < 1


def test_typographic_differences_still_count_as_recall():
    """Both sides go through the grounding checker's normalisation: a model
    reciting with straight quotes where the source has curly ones has recited."""
    rest = "\n".join(POEM["lines"][config.CONTAMINATION_PROMPT_LINES:]).upper()
    assert contamination.reproduction_rate(rest, POEM) == 1.0


def test_short_lines_are_excluded_from_scoring():
    """"And" or "O!" appear in almost any continuation by chance, and counting
    them would push every poem toward the threshold."""
    poem = {**POEM, "lines": ["opening line here", "and", "O!",
                              "a genuine line of some length"]}
    assert contamination.reproduction_rate("and O!", poem) == 0.0


# --- the flag -----------------------------------------------------------------

def test_threshold_is_not_perfect_recall():
    """Near-verbatim recall is still recall; a model reproducing four lines in
    five has plainly seen the poem."""
    assert config.CONTAMINATION_MATCH_THRESHOLD < 1.0
    assert contamination.is_memorised(config.CONTAMINATION_MATCH_THRESHOLD)


def test_low_recall_is_not_flagged():
    assert not contamination.is_memorised(0.1)


# --- summary and stratification -----------------------------------------------

def record(pid, rate):
    return {"poem_id": pid, "author": "A", "title": "T", "linecount": 5,
            "reproduction_rate": rate, "memorised": contamination.is_memorised(rate),
            "continuation": ""}


def test_summary_counts_memorised_poems():
    summary = contamination.summarise([record(1, 1.0), record(2, 0.0),
                                       record(3, 0.9), record(4, 0.1)])
    assert summary["n"] == 4
    assert summary["memorised"] == 2
    assert summary["memorised_share"] == 0.5


def test_stratification_splits_judge_scores_by_the_flag():
    """The reason the probe exists: a gap that survives on non-memorised poems
    is not recall."""
    from tests.test_judge import scored

    probes = [record(1, 1.0), record(2, 0.0)]
    scores = scored(1, 9, 1, 1) + scored(2, 8, 2, 2)
    split = contamination.stratify(probes, scores)

    assert split["memorised"]["n_pairs"] == 1
    assert split["not_memorised"]["n_pairs"] == 1
    assert split["memorised"]["grounding_gap"] == 8.0
    assert split["not_memorised"]["grounding_gap"] == 6.0


def test_stratification_ignores_poems_without_a_probe():
    from tests.test_judge import scored

    split = contamination.stratify([record(1, 1.0)],
                                   scored(1, 9, 1, 1) + scored(99, 5, 5, 5))
    assert split["memorised"]["n_pairs"] == 1
    assert split["not_memorised"]["n_pairs"] == 0


# --- the scoring window -------------------------------------------------------

def long_poem(n: int = 80) -> dict:
    """A poem far longer than the generation budget can reproduce."""
    return {"poem_id": 2, "author": "B", "title": "L", "linecount": n,
            "lines": [f"line number {i} of this rather long poem" for i in range(n)]}


def test_scoring_is_capped_to_a_fixed_window():
    """The regression that matters. Scoring the whole remainder made the
    threshold mechanically unreachable for long poems: 26.8% of the corpus
    could not have scored 0.8 however perfectly it recited, and reachability
    ran from 100% under 20 lines to 0.3% over 60. `memorised` would then have
    meant 'memorised AND short' — fatal for a flag whose only job is to
    stratify the headline results.
    """
    window = contamination.scored_lines(long_poem())
    assert len(window) == config.CONTAMINATION_SCORED_LINES


def test_a_long_poem_can_still_reach_the_threshold():
    """The property the window exists to restore: reciting the next few lines
    of an 80-line poem must be enough, since nothing longer fits the budget."""
    poem = long_poem()
    recited = "\n".join(contamination.scored_lines(poem))
    rate = contamination.reproduction_rate(recited, poem)
    assert rate == 1.0 and contamination.is_memorised(rate)


def test_the_window_fits_the_generation_budget():
    """A window the model has no room to emit would reintroduce the bug."""
    from src.data.filter import get_tokenizer

    tokenizer = get_tokenizer()
    text = "\n".join(contamination.scored_lines(long_poem()))
    assert (len(tokenizer(text)["input_ids"])
            < config.CONTAMINATION_MAX_NEW_TOKENS)


def test_the_window_starts_after_the_prompt():
    """Prompt lines are handed to the model, so scoring them would count the
    input as recall and flag every poem as memorised."""
    poem = long_poem()
    prompt = contamination.continuation_prompt(poem)
    assert not any(line in prompt for line in contamination.scored_lines(poem))
    assert contamination.reproduction_rate(prompt, poem) == 0.0


def test_short_poems_use_a_smaller_window():
    """Fewer qualifying lines than the window is legitimate — the corpus floor
    is 8 lines and the prompt takes 2 — but the denominator must shrink with
    it, not pad out with lines that do not exist."""
    window = contamination.scored_lines(POEM)
    assert len(window) == 3          # 5 lines, 2 taken as prompt
    assert all(line in POEM["lines"] for line in window)


def test_window_size_is_recorded_per_poem():
    """So a coarse denominator is visible rather than hidden in the aggregate."""
    summary = contamination.summarise([
        {"poem_id": 1, "reproduction_rate": 1.0, "memorised": True, "n_scored": 3},
        {"poem_id": 2, "reproduction_rate": 0.0, "memorised": False, "n_scored": 6},
    ])
    assert summary["short_window"] == 1
    assert summary["scored_lines"] == config.CONTAMINATION_SCORED_LINES
