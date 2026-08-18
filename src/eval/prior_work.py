"""Published reference values, and the table that compares ours against them.

The rubric asks for **comparisons between current and previous results**, not a
bibliography. Every constant below therefore carries three things: the value,
where it came from, and **the setting it was measured in**. The third is what
makes a comparison honest — Gudibande et al. studied 7-13B models imitating
ChatGPT judged by humans; this is a 0.5B model with a DeepSeek teacher judged
mechanically and by GPT-4o-mini. Those are not like-for-like, and a table that
silently implied they were would be worse than no table.

**Generated, never hand-typed.** Numbers copied into prose drift from the
results they came from the moment anything is re-run. :func:`build_comparison_table`
reads what was measured and writes the comparison, so the two cannot disagree.

**Direction comes from the data, never from the expectation.** A measured value
that contradicts published work is the more interesting outcome, and
:func:`verdict` has no way to prefer agreement — it compares an interval against
a value and reports what it finds. Where settings differ it returns
``not comparable`` rather than manufacturing a verdict.

Every quoted figure below must be checked against the source before it reaches
the report. None is here from memory alone; the page or section is recorded
beside it so a reader can verify it, and so can we.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Reference:
    """One published number, with everything needed to judge its relevance."""

    key: str
    #: What was reported. None where the source gives a direction rather than a
    #: figure — Gudibande et al. report a qualitative split, not a percentage.
    value: float | None
    #: Plain-language statement of the finding, in the source's own terms.
    finding: str
    #: The setting it was measured in. This is the field that decides whether a
    #: comparison is like-for-like, and it is never optional.
    setting: str
    citation: str
    #: Where in the source. Checked before the report quotes it.
    locator: str
    #: How our value relates to theirs, when the two are not directly
    #: comparable. Empty when they are.
    caveat: str = ""


#: Published values this project compares against.
#:
#: VERIFY EACH AGAINST THE SOURCE before the report quotes it. The locator field
#: says where to look. CLAUDE.md's rule is that no number reaches the writeup
#: from memory, and that applies to these more than to anything measured here.
REFERENCES: dict[str, Reference] = {
    "imitation_style_not_substance": Reference(
        key="imitation_style_not_substance",
        value=None,
        finding=("imitation models are adept at mimicking the teacher's style "
                 "but not its factuality; human raters scored them well because "
                 "they matched the teacher's form"),
        setting=("7-13B student models imitating ChatGPT across broad "
                 "instruction-following tasks, evaluated by human raters"),
        citation="Gudibande et al. 2023, arXiv:2305.15717",
        locator="abstract; §4",
        caveat=("qualitative, so the comparison is directional: does the same "
                "style/substance split appear at 0.5B with a mechanical "
                "grounding check in place of human raters?"),
    ),
    "low_rank_saturation": Reference(
        key="low_rank_saturation",
        value=None,
        finding=("adapting only a few weight matrices at a very small rank is "
                 "sufficient; increasing rank does not reliably improve "
                 "performance"),
        setting=("RoBERTa/DeBERTa on GLUE and GPT-3 175B on WikiSQL, MNLI and "
                 "SAMSum — classification and structured generation"),
        citation="Hu et al. 2021, arXiv:2106.09685",
        locator="§7.2, Table 6 (rank r in {1, 2, 4, 8, 64})",
        caveat=("their tasks have short, constrained outputs; ours is open "
                "generation scored by a judge"),
    ),
    "lima_sufficient_examples": Reference(
        key="lima_sufficient_examples",
        value=1000.0,
        finding="1,000 carefully curated examples sufficed for instruction alignment",
        setting=("a 65B LLaMa model, curated multi-domain instruction data, "
                 "human preference evaluation"),
        citation="Zhou et al. 2023, arXiv:2305.11206",
        locator="abstract; §5",
        caveat=("65B against 0.5B, and curated human-written data against "
                "synthetic teacher output; the data-size sweep straddles their "
                "threshold so the question is where our curve flattens relative "
                "to theirs"),
    ),
    "judge_position_consistency": Reference(
        key="judge_position_consistency",
        value=0.650,
        finding=("GPT-4 gave a consistent verdict on 65.0% of pairs when the "
                 "answer order was swapped; 30.0% favoured the first position "
                 "and 5.0% the second"),
        setting="GPT-4 as judge on MT-Bench pairwise comparisons",
        citation="Zheng et al. 2023, arXiv:2306.05685",
        locator="§3.3, Table 2",
        caveat=("a different judge model on a different task; ours is "
                "GPT-4o-mini scoring poem/interpretation pairs"),
    ),
    "judge_human_agreement": Reference(
        key="judge_human_agreement",
        value=0.800,
        finding=("strong LLM judges reached over 80% agreement with human "
                 "preferences, matching the level of agreement between humans"),
        setting="GPT-4 judge against human annotators on MT-Bench and Chatbot Arena",
        citation="Zheng et al. 2023, arXiv:2306.05685",
        locator="abstract; §4.2",
        caveat=("**not like-for-like.** No human annotation exists here, so our "
                "comparable number is agreement between two judges, or a "
                "judge's agreement with a known-correct pairing. Both are "
                "weaker evidence than judge-human agreement."),
    ),
    "evaluators_favour_own_output": Reference(
        key="evaluators_favour_own_output",
        value=None,
        finding=("LLM evaluators recognise their own generations and assign "
                 "them higher scores than human raters do"),
        setting="GPT-4, GPT-3.5 and Llama 2 evaluating summarisation output",
        citation="Panickssery et al. 2024, arXiv:2404.13076",
        locator="abstract; §3",
        caveat=("a design justification rather than a comparison — it is why "
                "student, teacher and both judges come from four different "
                "families"),
    ),
}


# --- verdicts -----------------------------------------------------------------

CONSISTENT = "consistent"
INCONSISTENT = "inconsistent"
NOT_COMPARABLE = "not comparable"


def verdict(measured: float | None, ci: tuple[float, float] | None,
            reference: Reference) -> str:
    """How a measured value relates to a published one.

    Three outcomes, and the third is the one that matters. ``not comparable``
    is returned whenever the reference is qualitative, the measurement is
    missing, or no interval was supplied — never a default of ``consistent``.
    A table that fell back to agreement when it had nothing to say would
    manufacture comparisons that were never made, which is exactly the failure
    a generated table is supposed to prevent.

    The test is whether the published value falls inside our interval. That is
    deliberately weak: a wide interval will contain almost anything, so
    ``consistent`` here means "our data does not contradict theirs", not "we
    replicated it". The interval is reported beside the verdict so a reader can
    see which they are looking at.
    """
    if reference.value is None:
        return NOT_COMPARABLE
    if measured is None or measured != measured:
        return NOT_COMPARABLE
    if ci is None or any(bound != bound for bound in ci):
        return NOT_COMPARABLE

    low, high = sorted(ci)
    return CONSISTENT if low <= reference.value <= high else INCONSISTENT


@dataclass
class Measurement:
    """One of our numbers, ready to be set against a published one."""

    name: str
    reference_key: str
    value: float | None = None
    ci: tuple[float, float] | None = None
    #: Free text where the comparison is directional rather than numeric —
    #: Gudibande et al. report a split, not a percentage.
    observation: str = ""
    judge: str | None = None


def compare(measurements: list[Measurement]) -> "object":
    """Join measured values against the published ones. Returns a DataFrame.

    Every row states the setting each number came from, because that is what
    lets a reader judge whether the comparison means anything. Where the
    settings differ in scale or task the wording is "consistent with" or
    "inconsistent with" — never "confirms" or "replicates".
    """
    import pandas as pd

    rows = []
    for measurement in measurements:
        reference = REFERENCES.get(measurement.reference_key)
        assert reference is not None, (
            f"{measurement.name} cites {measurement.reference_key!r}, which is "
            f"not in REFERENCES. A comparison against a value nobody recorded "
            f"is not a comparison.")

        rows.append({
            "measurement": measurement.name,
            "judge": measurement.judge or "",
            "our_value": measurement.value,
            "our_ci_low": (measurement.ci or (float("nan"),) * 2)[0],
            "our_ci_high": (measurement.ci or (float("nan"),) * 2)[1],
            "published_value": reference.value,
            "published_finding": reference.finding,
            "published_setting": reference.setting,
            "citation": reference.citation,
            "locator": reference.locator,
            "verdict": verdict(measurement.value, measurement.ci, reference),
            "caveat": reference.caveat,
            "observation": measurement.observation,
        })
    return pd.DataFrame(rows)


def build_comparison_table(measurements: list[Measurement], write: bool = True):
    """Build the comparison table and write it to ``results/``.

    Generated so the reported numbers cannot drift from the rest of
    ``results/``. Hand-typing them into prose is how a table ends up describing
    a run that no longer exists.
    """
    frame = compare(measurements)
    if write:
        config.PRIOR_WORK_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(config.PRIOR_WORK_CSV_PATH, index=False)
        log.info("prior-work comparison written to %s (%d rows)",
                 config.PRIOR_WORK_CSV_PATH, len(frame))

    counts = frame["verdict"].value_counts().to_dict()
    log.info("verdicts: %s", counts)
    return frame


def citations_used(frame) -> list[str]:
    """Distinct citations the table actually rests on.

    Each must be load-bearing in the body text, at the point where it justifies
    a decision or supplies a comparison value — not parked in a bibliography.
    """
    return sorted(frame["citation"].unique())
