"""Check whether an interpretation follows the required four-part schema.

Model-free and cheap, like :mod:`src.eval.grounding`. This measures *form*
only — whether the output has the shape it was asked for — and deliberately
says nothing about whether the content is any good.

That separation is the point. Gudibande et al. 2023 report that a distilled
model picks up its teacher's style but not its substance; H1 predicts format
compliance rises while H2 and H3 predict grounding does not. Those hypotheses
are only separable if form and substance are measured by different code.

The schema, from the teacher prompt in ``config.TEACHER_PROMPT_TEMPLATE``:

1. Central idea
2. Key images  (with exact quotes)
3. Tone
4. Interpretive claim
"""

from __future__ import annotations

import collections
import re

#: Expected part labels, in order. Matching is on the *number*, with the label
#: checked separately — a model that writes the four parts in order but names
#: part 3 "Mood" has followed the structure while drifting on wording, and
#: those are worth telling apart.
EXPECTED_LABELS: tuple[str, ...] = (
    "central idea",
    "key images",
    "tone",
    "interpretive claim",
)

#: A numbered part: start of a line, a digit, then '.' or ')'.
_PART = re.compile(r"^\s*(\d)\s*[.)]\s*(.*)$", re.MULTILINE)

#: The label is whatever precedes the first dash or colon on that line.
_LABEL = re.compile(r"^([^:\-—]{1,40})\s*[:\-—]")


def parse_parts(text: str) -> dict[int, str]:
    """Return ``{part number: text}`` for every numbered part found.

    A part's text runs to the start of the next numbered part, so multi-line
    parts survive intact.
    """
    matches = list(_PART.finditer(text))
    parts: dict[int, str] = {}

    for index, match in enumerate(matches):
        number = int(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        parts[number] = text[match.start(2):end].strip()

    return parts


def check(text: str) -> dict:
    """Audit one interpretation's structure.

    Returns ``compliant`` (all four parts present, numbered 1-4, in order and
    non-empty) alongside the detail needed to say *how* a failure failed:
    which parts are missing, which are empty, and which labels drifted.
    """
    parts = parse_parts(text)
    found = sorted(parts)

    missing = [n for n in (1, 2, 3, 4) if n not in parts]
    empty = [n for n, body in parts.items() if not body.strip()]
    extra = [n for n in found if n not in (1, 2, 3, 4)]

    mislabelled = []
    for number, expected in enumerate(EXPECTED_LABELS, start=1):
        body = parts.get(number, "")
        label_match = _LABEL.match(body)
        if label_match and expected not in label_match.group(1).lower():
            mislabelled.append(number)

    return {
        "compliant": not missing and not empty and found == [1, 2, 3, 4],
        "parts_found": found,
        "missing": missing,
        "empty": empty,
        "extra": extra,
        "mislabelled": mislabelled,
    }


def is_compliant(text: str) -> bool:
    """Return whether ``text`` follows the four-part schema."""
    return check(text)["compliant"]


def compliance_rate(interpretations: list[str]) -> float:
    """Fraction of interpretations that follow the schema."""
    if not interpretations:
        return 0.0
    return sum(is_compliant(text) for text in interpretations) / len(interpretations)


#: Words too common to say anything about a part's content.
_STOPWORDS = frozenset(
    "a an and the of to in is are with but or as its it this that for on at by "
    "from be been being was were has have had which who whom whose".split()
)


def part_vocabulary(interpretations: list[str], part: int,
                    drop_label: bool = True) -> collections.Counter:
    """Count the content words used in one part across many interpretations.

    Written for the tone slot, where the failure is invisible in every other
    metric: an interpretation can follow the schema perfectly and quote
    accurately while part 3 says "reflective and melancholy" every single time.
    That is template text in a slot the format check will happily pass, and a
    student trained on it learns the word rather than the judgement.

    A near-flat distribution means the teacher is reading each poem; one word
    dominating means it is not.
    """
    counter: collections.Counter = collections.Counter()

    for text in interpretations:
        body = parse_parts(text).get(part, "")
        if drop_label:
            body = _LABEL.sub("", body, count=1)
        words = re.findall(r"[a-z]+", body.lower())
        counter.update(w for w in words if w not in _STOPWORDS and len(w) > 2)

    return counter


def part_texts(interpretations: list[str], part: int) -> list[str]:
    """Return the raw text of one part from each interpretation."""
    return [parse_parts(text).get(part, "").strip() for text in interpretations]
