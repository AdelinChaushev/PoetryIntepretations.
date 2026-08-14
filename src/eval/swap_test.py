"""Build the matched / mismatched pairs the swap test scores.

The central measurement. One interpretation is scored against three poems:

``matched``
    the poem it was written for
``mismatched_random``
    a random poem by a **different** author — the standard control
``mismatched_same_author``
    a different poem by the **same** author — the strict control

Two gaps follow. **Grounding gap** = matched − mismatched_random. **Poem-level
gap** = matched − mismatched_same_author. Their difference is the author-level
component of apparent grounding: how much of the measured signal is recognising
the author rather than reading this poem.

This is a *relative* measurement, which is why it needs no ground truth. Judge
miscalibration moves every condition together and cancels in the difference.

**Every constraint here fails silently.** Nothing raises if a mismatched poem is
drawn from the same author, or if the "different poem by the same author" is the
same text under another title — the run completes and reports a plausible number
that means something else. The assertions at the bottom exist because the bug
would otherwise surface as a finding.
"""

from __future__ import annotations

import collections
import logging
import random
from typing import NamedTuple

import config

log = logging.getLogger(__name__)


class Pair(NamedTuple):
    """One interpretation shown to a judge alongside one poem."""

    #: The poem the interpretation was written FOR. Identifies the observation
    #: across conditions, so the three scores can be paired per poem.
    poem_id: int
    #: The poem actually SHOWN to the judge. Equal to ``poem_id`` only when the
    #: condition is ``matched``.
    shown_id: int
    condition: str
    interpretation: str
    #: Which arm produced the interpretation. ``teacher`` for the day-2
    #: validation run, otherwise the arm name.
    arm: str

    @property
    def is_matched(self) -> bool:
        return self.condition == "matched"


def _by_author(corpus: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for poem in corpus:
        grouped[poem["author"]].append(poem)
    return grouped


def build_pairs(
    interpretations: list[dict],
    corpus: list[dict],
    arm: str = "teacher",
    seed: int | None = None,
    near_duplicates: dict | None = None,
    strip_quotes: bool = False,
) -> list[Pair]:
    """Construct all three conditions for every interpretation.

    Args:
        interpretations: records with ``poem_id`` and ``interpretation``.
        corpus: the poems available to draw controls from.
        arm: which system produced these interpretations.
        seed: drawn deterministically so the same pairs are scored on every
            run — a judge call is paid for, and re-running with different
            controls would make two partial runs incomparable.
        strip_quotes: replace quoted spans with a placeholder, and suffix the
            arm name with ``_noquotes`` so the two never share a cache entry.
            This is the ablation that decides whether the judge is doing
            semantic work: the schema asks for exact quotations, so a matched
            pair contains literal substrings of the poem shown and a mismatched
            pair contains none — a judge could separate them by string matching
            alone, which the free checker already does. Separation that
            survives removal is the judge reading the interpretation's claims.
        near_duplicates: ``poem_id -> ids of same-author near-identical poems``,
            from :func:`src.data.filter.near_duplicate_ids`. Excluded from the
            same-author draw. Passing ``None`` disables the exclusion and is
            only right when the corpus is known to hold no duplicate titles.

    Returns:
        Three :class:`Pair` per interpretation, in ``config.SWAP_CONDITIONS``
        order. Interpretations whose author has no usable sibling are skipped
        with a warning rather than silently emitted with two conditions — a
        poem missing one condition would quietly drop out of the paired
        comparison and bias the gap toward whoever remained.
    """
    rng = random.Random(config.SEED if seed is None else seed)
    if strip_quotes:
        arm = f"{arm}_noquotes"
    by_id = {poem["poem_id"]: poem for poem in corpus}
    by_author = _by_author(corpus)
    authors = sorted(by_author)
    duplicates = near_duplicates or {}

    pairs: list[Pair] = []
    skipped = 0

    for record in sorted(interpretations, key=lambda r: r["poem_id"]):
        poem = by_id.get(record["poem_id"])
        if poem is None:
            continue

        siblings = [
            other for other in by_author[poem["author"]]
            if other["poem_id"] != poem["poem_id"]
            and other["poem_id"] not in duplicates.get(poem["poem_id"], ())
        ]
        if not siblings:
            skipped += 1
            continue

        # A different author entirely. Without this the standard control would
        # sometimes coincide with the strict one, and the two gaps would
        # converge for a reason that has nothing to do with the model.
        other_authors = [a for a in authors if a != poem["author"]]
        stranger = rng.choice(by_author[rng.choice(other_authors)])

        text = record["interpretation"]
        if strip_quotes:
            from src.eval.grounding import remove_quotes
            text = remove_quotes(text)
        pairs.extend([
            Pair(poem["poem_id"], poem["poem_id"], "matched", text, arm),
            Pair(poem["poem_id"], stranger["poem_id"],
                 "mismatched_random", text, arm),
            Pair(poem["poem_id"], rng.choice(siblings)["poem_id"],
                 "mismatched_same_author", text, arm),
        ])

    if skipped:
        log.warning("%d interpretations skipped: no same-author sibling that is "
                    "not a near-duplicate, so the strict condition is not "
                    "computable for them", skipped)
    log.info("built %d pairs for arm %r (%d interpretations x %d conditions)",
             len(pairs), arm, len(pairs) // len(config.SWAP_CONDITIONS),
             len(config.SWAP_CONDITIONS))
    return pairs


# --- assertions -------------------------------------------------------------
#
# Each of these fails silently into a plausible-looking result if omitted, which
# is the only reason they are assertions rather than tests alone.

def assert_all_conditions_present(pairs: list[Pair]) -> None:
    """Every poem has all three conditions, exactly once each."""
    seen: dict[int, list[str]] = collections.defaultdict(list)
    for pair in pairs:
        seen[pair.poem_id].append(pair.condition)

    for poem_id, conditions in seen.items():
        assert sorted(conditions) == sorted(config.SWAP_CONDITIONS), (
            f"poem {poem_id} has conditions {sorted(conditions)}, expected "
            f"{sorted(config.SWAP_CONDITIONS)} — a poem missing a condition "
            f"drops out of the paired comparison and biases the gap"
        )


def assert_never_scored_against_itself(pairs: list[Pair]) -> None:
    """Only ``matched`` shows the judge the poem the interpretation was for."""
    for pair in pairs:
        if pair.condition == "matched":
            assert pair.shown_id == pair.poem_id, (
                f"matched pair for poem {pair.poem_id} shows {pair.shown_id}")
        else:
            assert pair.shown_id != pair.poem_id, (
                f"{pair.condition} for poem {pair.poem_id} shows its OWN poem — "
                f"this silently turns a control into the matched condition")


def assert_random_is_different_author(pairs: list[Pair],
                                      corpus: list[dict]) -> None:
    """The standard control never draws the same author as the strict one."""
    by_id = {poem["poem_id"]: poem for poem in corpus}
    for pair in pairs:
        if pair.condition != "mismatched_random":
            continue
        assert by_id[pair.shown_id]["author"] != by_id[pair.poem_id]["author"], (
            f"mismatched_random for poem {pair.poem_id} drew the same author — "
            f"it would duplicate the same-author condition and the two gaps "
            f"would converge for the wrong reason")


def assert_same_author_is_a_different_poem(pairs: list[Pair],
                                           corpus: list[dict]) -> None:
    """The strict control draws the same author, never the same poem."""
    by_id = {poem["poem_id"]: poem for poem in corpus}
    for pair in pairs:
        if pair.condition != "mismatched_same_author":
            continue
        assert by_id[pair.shown_id]["author"] == by_id[pair.poem_id]["author"], (
            f"mismatched_same_author for poem {pair.poem_id} drew a different "
            f"author — that is the standard control, not the strict one")
        assert pair.shown_id != pair.poem_id


def assert_no_near_duplicate_controls(pairs: list[Pair],
                                      near_duplicates: dict) -> None:
    """The strict control is never a near-identical copy of the poem.

    PoetryDB publishes some poems under several titles. Drawing one as the
    "different poem by the same author" turns the strict condition into the
    matched condition, and the poem-level gap collapses to zero for a reason
    that has nothing to do with the model.
    """
    for pair in pairs:
        if pair.condition != "mismatched_same_author":
            continue
        assert pair.shown_id not in near_duplicates.get(pair.poem_id, ()), (
            f"strict control for poem {pair.poem_id} drew near-duplicate "
            f"{pair.shown_id} — this collapses the poem-level gap silently")


def assert_deterministic(interpretations: list[dict], corpus: list[dict],
                         **kwargs) -> None:
    """The same inputs produce the same pairs.

    Judge calls are paid for and runs are resumable, so two partial runs must
    agree on which controls were drawn or their scores cannot be pooled.
    """
    first = build_pairs(interpretations, corpus, **kwargs)
    second = build_pairs(interpretations, corpus, **kwargs)
    assert first == second, "pair construction is not deterministic"


def check_all(pairs: list[Pair], corpus: list[dict],
              near_duplicates: dict | None = None) -> None:
    """Run every structural assertion. Raises on the first violation."""
    assert_all_conditions_present(pairs)
    assert_never_scored_against_itself(pairs)
    assert_random_is_different_author(pairs, corpus)
    assert_same_author_is_a_different_poem(pairs, corpus)
    if near_duplicates is not None:
        assert_no_near_duplicate_controls(pairs, near_duplicates)
    log.info("swap-test pair construction passed every structural check")
