"""Tests for the quote checker.

This is tested hard because a silent bug here invalidates the whole project.
Too-forgiving matching scores invented lines as grounded and every arm looks
good; too-strict matching scores correct quotes as hallucinated and every arm
looks bad. Either way the headline number is wrong and nothing else notices.
"""

from __future__ import annotations

import pytest

from src.eval import grounding

POEM = {
    "title": "A Song of Autumn",
    "author": "Adam Lindsay Gordon",
    "lines": [
        "‘WHERE shall we go for our garlands glad",
        "At the falling of the year,",
        "When the burnt-up banks are yellow and sad,",
        "When the boughs are yellow and sere?",
    ],
    "linecount": 4,
}


# --- normalisation ---------------------------------------------------------

@pytest.mark.parametrize("variant", [
    "At the falling of the year",        # exact
    "at the falling of the year",        # case
    "At the falling of the year,",       # trailing punctuation
    "  At the falling of   the year  ",  # whitespace
    "At the falling of the year!",       # different punctuation
])
def test_presentation_differences_still_match(variant):
    """Typography must not decide whether a quote counts."""
    assert grounding.is_grounded(variant, POEM)


def test_curly_and_straight_quotes_are_equivalent():
    """The poem uses a curly opening quote; a model will emit a straight one."""
    assert grounding.is_grounded("'WHERE shall we go for our garlands glad", POEM)
    assert grounding.is_grounded("‘WHERE shall we go for our garlands glad", POEM)


def test_quote_spanning_a_line_break_matches():
    """Lines are joined before normalising, so a break is just whitespace."""
    spanning = "When the burnt-up banks are yellow and sad, When the boughs"
    assert grounding.is_grounded(spanning, POEM)


def test_em_dash_and_hyphen_are_equivalent():
    assert grounding.is_grounded("the burnt—up banks are yellow", POEM)


# --- the part that must NOT be forgiving -----------------------------------

def test_invented_line_is_not_grounded():
    assert not grounding.is_grounded("The sea was calm that silent night", POEM)


def test_plausible_but_wrong_wording_is_not_grounded():
    """Near-misses must fail. This is the case that decides the project."""
    assert not grounding.is_grounded("At the ending of the year", POEM)
    assert not grounding.is_grounded("When the boughs are golden and sere", POEM)


def test_reordered_words_are_not_grounded():
    assert not grounding.is_grounded("the year of the falling", POEM)


# --- quote extraction ------------------------------------------------------

def test_extracts_double_and_curly_quotes():
    text = ('The poet writes "At the falling of the year" and later '
            '“When the boughs are yellow and sere”.')
    assert grounding.extract_quotes(text) == [
        "At the falling of the year",
        "When the boughs are yellow and sere",
    ]


def test_apostrophes_are_not_treated_as_quotes():
    """Straight single quotes are apostrophes far more often than quote marks.

    Without this, "the sun's warmth ... it's clear" would be read as a quoted
    span and checked against the poem, producing a phantom hallucination.
    """
    text = "The sun's warmth fades and it's clear that autumn has arrived."
    assert grounding.extract_quotes(text) == []


def test_short_quotes_are_ignored():
    """A single common word is not a checkable claim about the poem."""
    assert grounding.extract_quotes('The poem dwells on "autumn" throughout.') == []


# --- whole-interpretation audit --------------------------------------------

def test_check_counts_grounded_and_hallucinated():
    text = ('It opens with "At the falling of the year" and closes with '
            '"The sea was calm that silent night".')
    result = grounding.check(text, POEM)

    assert result["n_quotes"] == 2
    assert result["n_grounded"] == 1
    assert result["hallucinated"] == ["The sea was calm that silent night"]
    assert result["grounded"] is False


def test_all_quotes_correct_is_grounded():
    text = ('Note "At the falling of the year" and "the boughs are yellow and sere".')
    result = grounding.check(text, POEM)

    assert result["n_quotes"] == 2
    assert result["grounded"] is True
    assert result["hallucinated"] == []


def test_interpretation_with_no_quotes_is_not_grounded():
    """Quoting nothing dodges the question; it must not score as a pass.

    Counting it as grounded would reward vague, unquotable output — precisely
    the failure mode this project exists to detect.
    """
    text = "The poem reflects on time and memory in a reflective, melancholy tone."
    result = grounding.check(text, POEM)

    assert result["n_quotes"] == 0
    assert result["grounded"] is False


def test_grounding_rate_over_pairs():
    good = 'It says "At the falling of the year".'
    bad = 'It says "The sea was calm that silent night".'

    assert grounding.grounding_rate([(good, POEM), (good, POEM)]) == 1.0
    assert grounding.grounding_rate([(good, POEM), (bad, POEM)]) == 0.5
    assert grounding.grounding_rate([]) == 0.0


# --- source markup ----------------------------------------------------------

def test_underscore_emphasis_does_not_break_a_correct_quote():
    """PoetryDB marks emphasis with underscores. `\\w` counts `_` as a word
    character, so a naive punctuation strip leaves it in and a verbatim quote
    fails the substring test — a false ungrounded verdict on 6.7% of poems."""
    poem = {"lines": ["May your fate be like _hers_ and unlike _mine_"]}
    interpretation = 'The close pleads: "may your fate be like hers and unlike mine"'
    assert grounding.check(interpretation, poem)["grounded"]


def test_underscore_stripping_does_not_glue_words_together():
    """Replaced with a space, not deleted — `a_b` is two words, not `ab`."""
    assert grounding.normalise("_hers_ and _mine_") == "hers and mine"
    assert grounding.normalise("fate_be") == "fate be"


def test_invented_line_is_still_ungrounded():
    """Normalisation must not rescue a quote the poem does not contain."""
    poem = {"lines": ["May your fate be like _hers_ and unlike _mine_"]}
    assert not grounding.check('It says "a line never written here"', poem)["grounded"]


# --- accents and ligatures ----------------------------------------------------

def test_metrical_accent_does_not_break_a_quote():
    """The accent in "Deservèd" marks scansion, not a different word."""
    poem = {"lines": ["Deservèd yet an end whose every part"]}
    assert grounding.is_grounded("Deserved yet an end", poem)


def test_hopkins_stress_marks_fold():
    poem = {"lines": ["That hére pérsonal tells off these heart-song powerful peals"]}
    assert grounding.is_grounded("that here personal tells off", poem)


def test_ligature_matches_its_expansion():
    poem = {"lines": ["O Cæsar, we who are about to die"]}
    assert grounding.is_grounded("O Caesar, we who are about to die", poem)


def test_folding_does_not_ground_a_different_word():
    poem = {"lines": ["Deservèd yet an end whose every part"]}
    assert not grounding.is_grounded("deserted yet an end", poem)
