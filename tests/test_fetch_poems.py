"""Tests for poem identity and deduplication.

Poem ids key the interpretations cache, the fold assignment shipped to Kaggle,
and every arm's output. An id that points at the wrong poem produces a wrong
grounding verdict with nothing raised, so the properties below are the ones
worth pinning down.
"""

from __future__ import annotations

from src.data import fetch_poems


def make(author: str, title: str, lines: list[str]) -> dict:
    return {"title": title, "author": author,
            "lines": lines, "linecount": len(lines)}


# --- numbering --------------------------------------------------------------

def test_ids_are_sequential_from_one():
    poems = fetch_poems.assign_ids([
        make("B", "x", ["one"]), make("A", "y", ["two"]), make("C", "z", ["three"]),
    ])
    assert sorted(p["poem_id"] for p in poems) == [1, 2, 3]


def test_numbering_does_not_depend_on_input_order():
    """Fetch order varies — cached authors are skipped, and the per-title
    fallback reorders others. Numbering by fetch order would renumber the
    corpus on a re-fetch and repoint every stored interpretation."""
    records = [make("A", "x", ["one"]), make("B", "y", ["two"]),
               make("C", "z", ["three"])]

    forward = {(p["author"], p["poem_id"]) for p in fetch_poems.assign_ids(records)}
    backward = {(p["author"], p["poem_id"])
                for p in fetch_poems.assign_ids(list(reversed(records)))}
    assert forward == backward


def test_same_title_by_same_author_gets_distinct_ids():
    """Blake wrote two "Holy Thursday", Poe two "To Helen". An id derived from
    author and title alone collided them onto one, and an interpretation could
    then be scored against a poem the teacher never saw."""
    poems = fetch_poems.assign_ids([
        make("William Blake", "Holy Thursday", ["Innocence version"]),
        make("William Blake", "Holy Thursday", ["Experience version"]),
    ])
    assert len({p["poem_id"] for p in poems}) == 2


# --- deduplication ----------------------------------------------------------

def test_identical_records_collapse():
    records = [make("A", "x", ["one"]), make("A", "x", ["one"])]
    assert len(fetch_poems._deduplicate(records)) == 1


def test_same_title_different_text_both_survive():
    """The case deduplication must NOT collapse — two distinct poems."""
    records = [make("A", "x", ["one"]), make("A", "x", ["two"])]
    assert len(fetch_poems._deduplicate(records)) == 2


def test_deduplication_keeps_unrelated_poems():
    records = [make("A", "x", ["one"]), make("A", "x", ["one"]),
               make("B", "y", ["two"])]
    assert len(fetch_poems._deduplicate(records)) == 2


# --- editorial line numbers ---------------------------------------------------

def numbered(n: int) -> list[str]:
    return [f"{i} line {i} of the poem" for i in range(1, n + 1)]


def test_ascending_line_numbers_are_stripped():
    poem = fetch_poems.strip_line_numbers(make("A", "x", numbered(6)))
    assert poem["lines"][0] == "line 1 of the poem"
    assert poem["lines"][5] == "line 6 of the poem"


def test_stripping_lets_a_couplet_quote_match():
    """The number sits between consecutive lines in the joined text, so a
    verbatim couplet fails the grounding check while it is there."""
    from src.eval import grounding

    raw = make("Henry Vaughan", "Peace",
               ["1 My Soul, there is a country",
                "2 Afar beyond the stars,",
                "3 Where stands a winged sentry",
                "4 All skillful in the wars;",
                "5 There, above noise and danger"])
    quote = "Where stands a winged sentry / All skillful in the wars"
    assert not grounding.is_grounded(quote, raw)
    assert grounding.is_grounded(quote, fetch_poems.strip_line_numbers(raw))


def test_poem_using_numerals_is_left_alone():
    """Real numerals do not count upward line by line — the ascent test is
    what separates an editorial apparatus from a poem about numbers."""
    lines = ["1914 was the year it began", "40 days and 40 nights",
             "7 swans a-swimming", "and then the war ended"]
    assert fetch_poems.strip_line_numbers(make("A", "x", lines))["lines"] == lines


def test_partially_numbered_poem_is_left_alone():
    lines = numbered(2) + ["an unnumbered line", "another unnumbered line",
                           "a third unnumbered line"]
    assert fetch_poems.strip_line_numbers(make("A", "x", lines))["lines"] == lines


def test_blank_stanza_breaks_do_not_defeat_detection():
    """Blank lines are unnumbered, so the numbers drift from list position —
    which is why detection tests ascent rather than index equality."""
    lines = numbered(4) + [""] + [f"{i} line {i} of the poem" for i in (5, 6)]
    assert fetch_poems.is_line_numbered(lines)


# --- mojibake -----------------------------------------------------------------

def test_cp1251_mojibake_is_repaired():
    """Latin-1 accents decoded as CP1251: 0xE8 prints as и instead of è."""
    poem = fetch_poems.fix_mojibake(make("A", "x", ["Deservиd yet an end"]))
    assert poem["lines"] == ["Deservèd yet an end"]


def test_hopkins_stress_marks_are_repaired():
    poem = fetch_poems.fix_mojibake(make("A", "x", ["That hйre pйrsonal tells off"]))
    assert poem["lines"] == ["That hére pérsonal tells off"]


def test_genuine_cyrillic_is_left_alone():
    """Only words mixing Cyrillic with Latin are repaired — that mixture is the
    corruption signature. A real Cyrillic word must survive untouched."""
    lines = ["и это по-русски", "спасибо"]
    assert fetch_poems.fix_mojibake(make("A", "x", lines))["lines"] == lines


def test_word_repaired_only_when_every_cyrillic_char_maps():
    """Byte 0xBE decodes to a Cyrillic letter but to '¾' in Latin-1; repairing
    a word into a fraction sign is worse than leaving it corrupt."""
    lines = ["Gaunt anapѕsts stand up out of the verse"]
    assert fetch_poems.fix_mojibake(make("A", "x", lines))["lines"] == lines
