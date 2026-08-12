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
