"""Turn raw poems and teacher interpretations into the training corpus.

Filtering is a funnel, and **the count at every stage is recorded**. A corpus
is not described by its final size alone; what was discarded, and why, says
more. The funnel table is a required figure for exactly that reason.

Two rules the rest of the pipeline depends on:

**Nothing is ever truncated.** A pair that does not fit ``MAX_SEQ_LEN`` is
dropped. Truncating would let the grounding checker match a quote against text
the model never saw, silently inflating every grounding number downstream.

**The teacher is audited with the student's checker.** The hallucination stage
calls :mod:`src.eval.grounding` — the same code the evaluation uses on model
output — so "the teacher is held to exactly the standard the student is held
to" is enforced rather than asserted.
"""

from __future__ import annotations

import logging
from typing import Callable, NamedTuple

import config
from src.eval import format_check, grounding

log = logging.getLogger(__name__)

_TOKENIZER = None


class Stage(NamedTuple):
    """One row of the funnel: what was checked, and what survived it."""

    name: str
    reason: str
    kept: int
    dropped: int


def get_tokenizer():
    """Load the student's tokeniser, caching it after the first call.

    Imported lazily and by name from ``config`` so that (a) the data pipeline
    does not pull in ``transformers`` until it genuinely needs to, and (b) the
    token budget is measured with the *real* tokeniser rather than estimated
    from word or line counts. Legal-length lines, archaic spelling and heavy
    punctuation all make estimates unreliable in this corpus.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:  # pragma: no cover - environment issue
            raise ImportError(
                "The token-length filter needs the real tokeniser.\n"
                "    pip install transformers\n"
                "This is a hard requirement: estimating token counts would let "
                "over-long pairs through, and those get silently truncated at "
                "training time, which corrupts the grounding measurement."
            ) from error
        _TOKENIZER = AutoTokenizer.from_pretrained(config.MODEL)
    return _TOKENIZER


def n_tokens(text: str) -> int:
    """Number of tokens ``text`` occupies for the student model."""
    return len(get_tokenizer().encode(text))


def pair_text(poem: dict, interpretation: str) -> str:
    """The full prompt + target string a training example becomes."""
    prompt = config.TEACHER_PROMPT_TEMPLATE.format(
        title=poem["title"],
        author=poem["author"],
        poem="\n".join(poem["lines"]),
        min_words=config.MIN_WORDS,
        max_words=config.MAX_WORDS,
    )
    return prompt + interpretation


# --- individual filters ----------------------------------------------------

def poem_tokens(poem: dict) -> int:
    """Tokens the poem text alone occupies."""
    return n_tokens("\n".join(poem["lines"]))


def within_length_bounds(poem: dict) -> bool:
    """Poem is long enough to quote from and short enough to train on.

    The lower bound is in lines, because it is about having enough *distinct*
    lines to quote two or three without reproducing the poem. The upper bound
    is in tokens, because that is the constraint that actually exists — line
    count is a poor proxy for it, since line length varies by a factor of
    several across this corpus.
    """
    return (poem["linecount"] >= config.MIN_LINES
            and poem_tokens(poem) <= config.MAX_POEM_TOKENS)


def within_word_bounds(interpretation: str) -> bool:
    """Interpretation is neither too thin to be one nor padded."""
    return config.MIN_WORDS <= len(interpretation.split()) <= config.MAX_WORDS


def fits_context(poem: dict, interpretation: str) -> bool:
    """Prompt plus target fits the sequence budget without truncation."""
    return n_tokens(pair_text(poem, interpretation)) <= config.MAX_SEQ_LEN


# --- the funnel ------------------------------------------------------------

def build_corpus(
    raw_poems: list[dict],
    interpretations: list[dict],
    check_tokens: bool = True,
) -> tuple[list[dict], list[Stage]]:
    """Filter raw poems and interpretations into the training corpus.

    Args:
        raw_poems: records from :mod:`src.data.fetch_poems`.
        interpretations: records with ``poem_id`` and ``interpretation``.
        check_tokens: run the token-budget stage. Only ever disabled in tests,
            where loading a tokeniser would dominate the runtime.

    Returns:
        The surviving corpus, and the funnel as a list of :class:`Stage`.
        Every stage is recorded even when it drops nothing — a stage missing
        from the table is indistinguishable from a stage that was never run.
    """
    by_id = {record["poem_id"]: record["interpretation"]
             for record in interpretations}

    funnel: list[Stage] = []
    surviving = list(raw_poems)
    funnel.append(Stage("fetched", "retrieved from PoetryDB",
                        len(surviving), 0))

    def apply(name: str, reason: str,
              predicate: Callable[[dict], bool]) -> None:
        nonlocal surviving
        before = len(surviving)
        surviving = [poem for poem in surviving if predicate(poem)]
        funnel.append(Stage(name, reason, len(surviving), before - len(surviving)))

    apply("length bounds",
          f"under {config.MIN_LINES} lines, or poem over "
          f"{config.MAX_POEM_TOKENS} tokens",
          within_length_bounds)

    apply("has interpretation",
          "teacher produced no output for this poem",
          lambda poem: poem["poem_id"] in by_id)

    apply("schema followed",
          "interpretation missing one of the four parts",
          lambda poem: format_check.is_compliant(by_id[poem["poem_id"]]))

    apply("word bounds",
          f"interpretation outside [{config.MIN_WORDS}, {config.MAX_WORDS}] words",
          lambda poem: within_word_bounds(by_id[poem["poem_id"]]))

    apply("quotes grounded",
          "teacher quoted lines absent from the poem (hallucination)",
          lambda poem: grounding.check(by_id[poem["poem_id"]], poem)["grounded"])

    if check_tokens:
        apply("fits context",
              f"prompt + target over {config.MAX_SEQ_LEN} tokens "
              f"(dropped, never truncated)",
              lambda poem: fits_context(poem, by_id[poem["poem_id"]]))

    corpus = [
        {**poem, "interpretation": by_id[poem["poem_id"]]}
        for poem in surviving
    ]

    if config.N_POEMS is not None:
        corpus = corpus[:config.N_POEMS]
        funnel.append(Stage("corpus cap", f"capped at N_POEMS={config.N_POEMS}",
                            len(corpus), len(surviving) - len(corpus)))

    log.info("corpus: %d poems from %d raw", len(corpus), len(raw_poems))
    return corpus, funnel


def funnel_table(funnel: list[Stage]):
    """Return the funnel as a DataFrame — Figure 1."""
    import pandas as pd

    start = funnel[0].kept
    return pd.DataFrame([
        {
            "stage": stage.name,
            "dropped": stage.dropped,
            "kept": stage.kept,
            "% of raw": f"{stage.kept / start:.1%}" if start else "-",
            "reason for dropping": stage.reason,
        }
        for stage in funnel
    ])


# --- teacher quality -------------------------------------------------------

def wilson_interval(successes: int, total: int,
                    z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves sensibly when the rate is near 0 or 1 — which is exactly where a
    hallucination rate is likely to sit.
    """
    if total == 0:
        return (0.0, 0.0)

    phat = successes / total
    denominator = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denominator
    margin = (z * ((phat * (1 - phat) / total
                    + z**2 / (4 * total**2)) ** 0.5)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def hallucination_rate(
    interpretations: list[dict],
    poems: list[dict],
) -> tuple[float, tuple[float, float]]:
    """Share of teacher interpretations quoting lines absent from their poem.

    Reported rather than hidden, for two reasons. It is the honest description
    of the training data — the targets are synthetic, and this is how good they
    are. And it is the reference point for every later grounding number: a
    student that quotes as accurately as its teacher has learned what it was
    shown, while one that quotes less accurately has learned the format and
    dropped the substance.
    """
    by_id = {poem["poem_id"]: poem for poem in poems}

    checked = [
        grounding.check(record["interpretation"], by_id[record["poem_id"]])
        for record in interpretations
        if record["poem_id"] in by_id
    ]
    if not checked:
        return 0.0, (0.0, 0.0)

    hallucinated = sum(not result["grounded"] for result in checked)
    return hallucinated / len(checked), wilson_interval(hallucinated, len(checked))
