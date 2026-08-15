"""Tests for the filtering funnel.

The funnel decides what the model trains on and what every reported count
means. The failure that matters most is a *silent* one: a stage that drops
nothing because its predicate is wrong reads exactly like a stage where
everything passed.
"""

from __future__ import annotations

import config
from src.data import filter as data_filter

POEM_LINES = [
    "‘WHERE shall we go for our garlands glad",
    "At the falling of the year,",
    "When the burnt-up banks are yellow and sad,",
    "When the boughs are yellow and sere?",
]


def make_poem(poem_id: str, linecount: int = 12) -> dict:
    return {
        "poem_id": poem_id,
        "title": f"Poem {poem_id}",
        "author": "Adam Lindsay Gordon",
        "lines": POEM_LINES,
        "linecount": linecount,
    }


def make_interpretation(poem_id: str, body: str | None = None) -> dict:
    if body is None:
        body = (
            '1. Central idea: The poem watches a year turn and asks what is '
            'left when it does, returning again and again to loss.\n'
            '2. Key images: "At the falling of the year" fixes the season, '
            'and "When the boughs are yellow and sere" gives the colour of it.\n'
            '3. Tone: Elegiac and questioning.\n'
            '4. Interpretive claim: The repeated questions are not rhetorical; '
            'the speaker genuinely does not know where the old garlands went, '
            'and the poem refuses to supply an answer for them anywhere here.\n'
        )
    return {"poem_id": poem_id, "interpretation": body}


def test_clean_pair_survives_every_stage():
    poems = [make_poem("a")]
    interps = [make_interpretation("a")]

    corpus, funnel = data_filter.build_corpus(poems, interps, check_tokens=False)

    assert len(corpus) == 1
    assert corpus[0]["interpretation"] == interps[0]["interpretation"]
    assert all(stage.dropped == 0 for stage in funnel)


def test_every_stage_appears_even_when_it_drops_nothing():
    """A stage missing from the table is indistinguishable from one not run."""
    corpus, funnel = data_filter.build_corpus(
        [make_poem("a")], [make_interpretation("a")], check_tokens=False
    )
    names = [stage.name for stage in funnel]

    expected = ["fetched", "length bounds", "has interpretation",
                "schema followed", "word bounds", "quotes grounded"]
    # SMOKE sets N_POEMS, which adds a cap stage. The stage genuinely exists in
    # that mode, so the expectation depends on the config rather than on which
    # mode the suite happened to be run in.
    if config.N_POEMS is not None:
        expected.append("corpus cap")

    assert names == expected


def test_poem_outside_length_bounds_is_dropped():
    """Too few lines to quote from, or too many tokens to train on."""
    short = make_poem("short", linecount=config.MIN_LINES - 1)
    short["lines"] = POEM_LINES[:2]

    huge = make_poem("huge")
    huge["lines"] = ["word " * 300] * 20      # well past MAX_POEM_TOKENS
    huge["linecount"] = 20

    poems = [short, huge, make_poem("ok")]
    interps = [make_interpretation(p["poem_id"]) for p in poems]

    corpus, funnel = data_filter.build_corpus(poems, interps, check_tokens=False)

    assert [p["poem_id"] for p in corpus] == ["ok"]
    assert next(s for s in funnel if s.name == "length bounds").dropped == 2


def test_poem_without_an_interpretation_is_dropped():
    corpus, funnel = data_filter.build_corpus(
        [make_poem("a"), make_poem("b")], [make_interpretation("a")],
        check_tokens=False,
    )

    assert [p["poem_id"] for p in corpus] == ["a"]
    assert next(s for s in funnel if s.name == "has interpretation").dropped == 1


def test_interpretation_missing_a_part_is_dropped():
    broken = ('1. Central idea: Something.\n'
              '2. Key images: "At the falling of the year".\n'
              '3. Tone: Elegiac.\n')  # no part 4

    corpus, funnel = data_filter.build_corpus(
        [make_poem("a")], [make_interpretation("a", broken)], check_tokens=False,
    )

    assert corpus == []
    assert next(s for s in funnel if s.name == "schema followed").dropped == 1


def test_hallucinated_quote_is_dropped():
    """The stage that audits the teacher with the student's own checker."""
    # Long enough to clear the word-bound stage, so this test isolates the
    # grounding stage rather than being dropped earlier for an unrelated reason.
    invented = (
        '1. Central idea: The poem is about the sea at night and the way a '
        'watcher on the shore measures time by the movement of the water '
        'rather than by any clock, which is what gives it its stillness.\n'
        '2. Key images: "The sea was calm that silent night" sets the scene, '
        'and "no gull disturbed the water" holds it there without motion.\n'
        '3. Tone: Still and watchful throughout, with an undertow of dread.\n'
        '4. Interpretive claim: The stillness is a held breath rather than '
        'peace, and the poem never lets it break, which is what gives the '
        'closing lines their particular unease and leaves the reader waiting '
        'for a resolution that the poem quite deliberately declines to give.\n'
    )

    corpus, funnel = data_filter.build_corpus(
        [make_poem("a")], [make_interpretation("a", invented)], check_tokens=False,
    )

    assert corpus == []
    assert next(s for s in funnel if s.name == "quotes grounded").dropped == 1


def test_interpretation_outside_word_bounds_is_dropped():
    too_short = ('1. Central idea: Autumn.\n'
                 '2. Key images: "At the falling of the year".\n'
                 '3. Tone: Sad.\n'
                 '4. Interpretive claim: It ends.\n')

    corpus, funnel = data_filter.build_corpus(
        [make_poem("a")], [make_interpretation("a", too_short)], check_tokens=False,
    )

    assert corpus == []
    assert next(s for s in funnel if s.name == "word bounds").dropped == 1


def test_funnel_counts_are_internally_consistent():
    """kept + dropped at each stage must equal the previous stage's kept."""
    poems = [make_poem(str(i)) for i in range(5)]
    poems[0]["linecount"] = 1                       # fails line bounds
    interps = [make_interpretation(p["poem_id"]) for p in poems[:4]]  # 1 missing

    _, funnel = data_filter.build_corpus(poems, interps, check_tokens=False)

    for previous, stage in zip(funnel, funnel[1:]):
        assert stage.kept + stage.dropped == previous.kept, stage.name


# --- Wilson interval -------------------------------------------------------

def test_wilson_interval_brackets_the_estimate():
    low, high = data_filter.wilson_interval(20, 100)
    assert low < 0.20 < high


def test_wilson_interval_stays_in_range_at_the_extremes():
    """Where the normal approximation would give an impossible interval."""
    assert data_filter.wilson_interval(0, 50)[0] >= 0.0
    assert data_filter.wilson_interval(50, 50)[1] <= 1.0


def test_wilson_interval_narrows_with_more_data():
    narrow = data_filter.wilson_interval(200, 1000)
    wide = data_filter.wilson_interval(2, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_hallucination_rate_counts_ungrounded_interpretations():
    poems = [make_poem("a"), make_poem("b")]
    invented = ('1. Central idea: The sea.\n'
                '2. Key images: "The sea was calm that silent night".\n'
                '3. Tone: Still.\n'
                '4. Interpretive claim: It never breaks.\n')

    rate, (low, high) = data_filter.hallucination_rate(
        [make_interpretation("a"), make_interpretation("b", invented)], poems,
    )

    assert rate == 0.5
    assert low < 0.5 < high


# --- the derived length bound ----------------------------------------------

def test_prompt_overhead_constant_matches_the_real_tokeniser():
    """MAX_POEM_TOKENS is derived from this constant, so it must stay true.

    If the prompt template is edited without updating PROMPT_OVERHEAD_TOKENS,
    the poem budget silently drifts and pairs start failing the token check for
    reasons nobody can see in the funnel.
    """
    empty = {"title": "T", "author": "A", "lines": [], "linecount": 0}
    measured = data_filter.n_tokens(
        data_filter.pair_text(empty, "word " * config.MAX_WORDS)
    )
    assert abs(measured - config.PROMPT_OVERHEAD_TOKENS) <= 16, (
        f"prompt overhead is {measured} tokens but config says "
        f"{config.PROMPT_OVERHEAD_TOKENS}; update it and MAX_POEM_TOKENS"
    )


def test_length_bound_rejects_on_tokens_not_lines():
    """A long poem of short lines is fine; a short poem of long lines is not.

    This is the whole reason the line cap was removed: line count was rejecting
    on the wrong variable and discarding ~4% of usable poems.
    """
    # Sized RELATIVE to the budget, not to a number that happens to suit the
    # full-mode config. SMOKE shrinks MAX_POEM_TOKENS from 1632 to 96, and a
    # fixture with hardcoded lengths tests nothing there.
    budget = config.MAX_POEM_TOKENS
    many_short_lines = {"poem_id": "a", "title": "t", "author": "x",
                        "lines": ["oh"] * (budget // 4),
                        "linecount": budget // 4}
    few_long_lines = {"poem_id": "b", "title": "t", "author": "x",
                      "lines": ["word " * budget] * 12, "linecount": 12}

    assert many_short_lines["linecount"] > few_long_lines["linecount"], (
        "the fixture must have MORE lines in the accepted poem, or it does not "
        "demonstrate that rejection is on tokens rather than lines")
    assert data_filter.within_length_bounds(many_short_lines)
    assert not data_filter.within_length_bounds(few_long_lines)


def test_poem_below_min_lines_is_rejected():
    tiny = {"poem_id": "c", "title": "t", "author": "x",
            "lines": ["a line"] * (config.MIN_LINES - 1),
            "linecount": config.MIN_LINES - 1}
    assert not data_filter.within_length_bounds(tiny)


# --- near-duplicate detection -------------------------------------------------

def poem(pid, author, title, lines):
    return {"poem_id": pid, "author": author, "title": title,
            "lines": lines, "linecount": len(lines)}


SONNET = ["Now God be thanked who has matched us with his hour",
          "And caught our youth and wakened us from sleeping",
          "With hand made sure clear eye and sharpened power"]


def test_same_text_under_two_titles_is_flagged():
    """PoetryDB publishes Brooke's "The Soldier" under three titles, and
    deduplication keys on title as well as text, so they survive it."""
    found = data_filter.near_duplicate_ids([
        poem(1, "Rupert Brooke", "The Soldier", SONNET),
        poem(2, "Rupert Brooke", "1914 V: The Soldier", SONNET),
    ])
    assert found == {1: {2}, 2: {1}}


def test_different_poems_by_one_author_are_not_flagged():
    found = data_filter.near_duplicate_ids([
        poem(1, "A", "x", SONNET),
        poem(2, "A", "y", ["A slumber did my spirit seal",
                           "I had no human fears she seemed",
                           "A thing that could not feel the touch"]),
    ])
    assert found == {}


def test_identical_text_by_different_authors_is_not_compared():
    """Only same-author pairs are compared. Two poets writing near-identical
    text does not occur here, and restricting keeps this out of O(n^2)."""
    assert data_filter.near_duplicate_ids([
        poem(1, "A", "x", SONNET), poem(2, "B", "y", SONNET),
    ]) == {}


def test_single_poem_corpus_is_safe():
    assert data_filter.near_duplicate_ids([poem(1, "A", "x", SONNET)]) == {}
