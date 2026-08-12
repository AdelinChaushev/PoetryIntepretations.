"""Check whether quoted lines actually appear in the poem they cite.

This is the project's cheapest and most trustworthy measurement: no model, no
judge, no opinion. An interpretation claims the poem contains a line; either it
does or it does not.

The same functions are used in two places, deliberately:

* :mod:`src.data.filter` audits the *teacher* with them, to measure the
  hallucination rate of the training targets
* the evaluation audits the *student* with them

Holding both to one implementation is what makes "the teacher is held to
exactly the standard the student is held to" a fact rather than a claim.

Matching is deliberately forgiving about *presentation* and strict about
*content*. A model that writes ``"Where shall we go for our garlands glad"``
when the poem reads ``'WHERE shall we go for our garlands glad`` has quoted
correctly; only the curly quote and the capitalisation differ. A model that
invents a line has not, and no amount of normalisation should rescue it.
"""

from __future__ import annotations

import re
import unicodedata

#: Characters that vary by typography without changing what was said.
#: Curly quotes, dashes and ellipses are the ones poetry sources actually
#: differ on — PoetryDB uses curly single quotes where a model will emit
#: straight ones.
_EQUIVALENTS = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "…": "...", " ": " ",
}
_TRANSLATION = str.maketrans(_EQUIVALENTS)

_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")

#: Quoted spans: straight or curly double quotes, or curly singles. Straight
#: single quotes are NOT treated as quote marks — they are apostrophes far more
#: often than not in this corpus ("o'er", "'tis", "sun's").
_QUOTED = re.compile(r'"([^"]{%d,})"|“([^”]{%d,})”|‘([^’]{%d,})’'
                     % (3, 3, 3))


def normalise(text: str) -> str:
    """Reduce text to a form where only wording differences remain.

    Unicode is folded to a canonical form, typographic variants are unified,
    case is dropped, punctuation is removed, and runs of whitespace — including
    the line breaks that make a quoted couplet span two lines — collapse to a
    single space.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_TRANSLATION)
    text = text.lower()
    text = _PUNCTUATION.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def extract_quotes(interpretation: str, min_words: int = 3) -> list[str]:
    """Return the quoted spans in ``interpretation``.

    Spans shorter than ``min_words`` are ignored: a model quoting a single
    common word ("death") is not making a checkable claim about the poem, and
    counting it would inflate the grounding rate with trivially-true matches.
    """
    quotes = []
    for match in _QUOTED.finditer(interpretation):
        span = next(group for group in match.groups() if group is not None)
        if len(span.split()) >= min_words:
            quotes.append(span.strip())
    return quotes


def poem_text(poem: dict) -> str:
    """Return a poem's lines joined into one normalised string.

    Joined *before* normalising so a quote spanning a line break still matches:
    the break becomes a space like any other whitespace.
    """
    return normalise(" ".join(poem["lines"]))


def is_grounded(quote: str, poem: dict | str) -> bool:
    """Return whether ``quote`` appears in the poem."""
    haystack = poem if isinstance(poem, str) else poem_text(poem)
    return normalise(quote) in haystack


def check(interpretation: str, poem: dict) -> dict:
    """Audit every quote in an interpretation against its poem.

    Returns a dict with ``n_quotes``, ``n_grounded``, the ``hallucinated``
    quotes themselves, and ``grounded`` — True only when the interpretation
    made at least one checkable claim and every one of them held up.

    An interpretation that quotes nothing is **not** grounded. It has dodged
    the question rather than answered it, and treating it as a pass would
    reward exactly the vague, unquotable output this project exists to detect.
    """
    haystack = poem_text(poem)
    quotes = extract_quotes(interpretation)
    hallucinated = [q for q in quotes if not is_grounded(q, haystack)]

    return {
        "n_quotes": len(quotes),
        "n_grounded": len(quotes) - len(hallucinated),
        "hallucinated": hallucinated,
        "grounded": bool(quotes) and not hallucinated,
    }


def grounding_rate(pairs: list[tuple[str, dict]]) -> float:
    """Fraction of (interpretation, poem) pairs that are fully grounded."""
    if not pairs:
        return 0.0
    return sum(check(text, poem)["grounded"] for text, poem in pairs) / len(pairs)
