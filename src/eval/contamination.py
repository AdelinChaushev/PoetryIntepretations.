"""Measure how much of the corpus the base model can already recite.

Public-domain poetry is public domain because it is old and widely reproduced,
which is exactly what puts it in web-scale pretraining data. Qwen has very
likely seen these poems — and commentary about them — before this project starts.

**That cannot be avoided, so it is measured.** Contemporary poetry outside
pretraining data is under copyright and cannot be sent to an API, so switching
corpus is not an option. Author-level folds prevent leakage through *our*
fine-tuning data and do nothing about leakage that arrived during pretraining.

The probe: give the base model the first few lines of a poem and ask it to
continue, with **greedy decoding**. If it reproduces the next
``CONTAMINATION_SCORED_LINES`` lines, it memorised the poem rather than read it.

**Recall is scored over a fixed window, not the whole poem**, and that is a
correction rather than a simplification. Against a fixed generation budget,
scoring the entire remainder made the threshold mechanically unreachable for
long poems — 26.8% of the corpus could not have scored 0.8 however perfectly it
recited — so ``memorised`` would have encoded poem length as much as recall.

The output is a per-poem ``memorised`` flag, and the point is not the headline
number but the **stratification**. If the grounding gap is similar on memorised
and non-memorised poems, recall is not driving the result. If it collapses on
non-memorised poems, that is a finding, and an important one — it would mean the
measurement is reading the model's memory rather than its comprehension.

Runs on Kaggle, in the same session as training, because it needs the base model
and the model never comes down. No judge calls.
"""

from __future__ import annotations

import json
import logging

import config
from src.eval import grounding

log = logging.getLogger(__name__)


def continuation_prompt(poem: dict, n_lines: int | None = None) -> str:
    """The opening lines the model is asked to continue.

    Deliberately bare — no title, no author, no instruction. Naming the poem
    would let a model that recognises the *title* recite from that alone, which
    is a different and weaker claim than recognising the text. What is being
    measured is whether the opening lines are enough to retrieve the rest.
    """
    n = config.CONTAMINATION_PROMPT_LINES if n_lines is None else n_lines
    return "\n".join(poem["lines"][:n])


def scored_lines(poem: dict, n_lines: int | None = None,
                 k: int | None = None) -> list[str]:
    """The lines recall is scored over: a fixed window, not the whole poem.

    **The window is what makes the flag mean one thing corpus-wide.** Scoring
    the entire remainder against a fixed generation budget made the threshold
    mechanically unreachable for long poems — at 256 tokens, 26.8% of the
    corpus could not have reached 0.8 however perfectly it recited, and
    reachability ran from 100% under 20 lines to 0.3% over 60. ``memorised``
    would then have meant "memorised *and* short", which is fatal for a flag
    whose only job is to stratify the headline results.

    Short lines are excluded first. "And" or "O!" appear in almost any
    continuation by chance, and counting them would push every poem toward the
    threshold.
    """
    n = config.CONTAMINATION_PROMPT_LINES if n_lines is None else n_lines
    width = config.CONTAMINATION_SCORED_LINES if k is None else k
    qualifying = [line for line in poem["lines"][n:]
                  if len(line.split()) >= config.CONTAMINATION_MIN_LINE_WORDS]
    return qualifying[:width]


def reproduction_rate(generated: str, poem: dict,
                      n_lines: int | None = None, k: int | None = None) -> float:
    """Share of the scored window that appears in the generated text.

    Line-level rather than a similarity score, because the claim being tested is
    verbatim recall: either a line was reproduced or it was not. Both sides go
    through the same normalisation the grounding checker uses, so typographic
    differences do not count as failures to recall — the model reciting with
    straight quotes where the source has curly ones has still recited.
    """
    window = scored_lines(poem, n_lines, k)
    if not window:
        return 0.0

    haystack = grounding.normalise(generated)
    # The truthiness guard is load-bearing: "" is a substring of every string,
    # so a line that normalises away would otherwise count as a perfect hit.
    hits = sum(1 for line in window
               if grounding.normalise(line) and grounding.normalise(line) in haystack)
    return hits / len(window)


def is_memorised(rate: float) -> bool:
    """Whether recall counts as memorisation.

    Not 1.0: near-verbatim recall with small variations is still recall, and a
    model that reproduces four lines in five has plainly seen the poem. The
    threshold is a judgement call and is stated rather than hidden.
    """
    return rate >= config.CONTAMINATION_MATCH_THRESHOLD


def generate_continuation(poem: dict, model, tokenizer,
                          max_new_tokens: int | None = None) -> str:
    """Greedy continuation of a poem's opening lines.

    **Greedy, never sampled.** Sampling would let a model that half-remembers a
    poem miss it by luck, and let one that does not remember stumble into it.
    The question is whether the continuation is *retrievable*, which is a
    property of the model, not of a random draw.
    """
    import torch

    prompt = continuation_prompt(poem)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    budget = max_new_tokens or config.CONTAMINATION_MAX_NEW_TOKENS

    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=budget,
            do_sample=False,                 # greedy
            temperature=None, top_p=None,    # unset, or transformers warns
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[0][encoded["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def probe(poem: dict, model, tokenizer) -> dict:
    """Probe one poem. Returns the record written to the cache."""
    continuation = generate_continuation(poem, model, tokenizer)
    window = scored_lines(poem)
    rate = reproduction_rate(continuation, poem)
    return {
        "poem_id": poem["poem_id"],
        "author": poem["author"],
        "title": poem["title"],
        "linecount": poem["linecount"],
        "reproduction_rate": round(rate, 4),
        "memorised": is_memorised(rate),
        # Recorded so the window is auditable rather than assumed. Poems with
        # fewer qualifying lines than the window are scored on a smaller
        # denominator, which makes their rate coarser — visible here, not
        # hidden in an aggregate.
        "n_scored": len(window),
        "continuation": continuation[:200],
    }


def load_cached() -> list[dict]:
    """Probe results already on disk, newest per poem winning."""
    path = config.CONTAMINATION_PATH
    if not path.exists():
        return []
    by_id: dict = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            by_id[record["poem_id"]] = record
    return list(by_id.values())


def probe_all(poems: list[dict], model, tokenizer) -> list[dict]:
    """Probe every poem, skipping any already done.

    Resumable for the same reason generation is: this runs on Kaggle, sessions
    die, and a probe that has to restart from zero will not get finished.
    """
    path = config.CONTAMINATION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {record["poem_id"] for record in load_cached()}
    pending = [poem for poem in poems if poem["poem_id"] not in done]
    log.info("contamination probe: %d cached, %d to run",
             len(poems) - len(pending), len(pending))

    try:
        from tqdm.auto import tqdm
        pending_iter = tqdm(pending, unit="poem", desc="contamination")
    except ImportError:  # pragma: no cover
        pending_iter = pending

    with path.open("a", encoding="utf-8") as handle:
        for poem in pending_iter:
            record = probe(poem, model, tokenizer)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    wanted = {poem["poem_id"] for poem in poems}
    return [r for r in load_cached() if r["poem_id"] in wanted]


def summarise(records: list[dict]) -> dict:
    """The reportable number, and the ingredients of the stratification."""
    if not records:
        return {"n": 0}
    rates = sorted(r["reproduction_rate"] for r in records)
    memorised = [r for r in records if r["memorised"]]
    short = [r for r in records
             if r.get("n_scored", config.CONTAMINATION_SCORED_LINES)
             < config.CONTAMINATION_SCORED_LINES]
    return {
        "n": len(records),
        "memorised": len(memorised),
        "memorised_share": len(memorised) / len(records),
        "median_reproduction": rates[len(rates) // 2],
        "mean_reproduction": sum(rates) / len(rates),
        "threshold": config.CONTAMINATION_MATCH_THRESHOLD,
        "scored_lines": config.CONTAMINATION_SCORED_LINES,
        # Poems with a short window are scored on a coarser denominator, so
        # the share is worth seeing next to the headline rather than buried.
        "short_window": len(short),
    }


def stratify(records: list[dict], scores: list[dict]) -> dict:
    """Split a judge's per-poem gaps by whether the poem was memorised.

    **The whole reason the probe exists.** A grounding gap that survives on
    non-memorised poems is not recall. One that collapses there is, and that
    would change what every other number in the project means.

    Takes judge records rather than a summary so the split happens on the same
    paired per-poem differences the bootstrap uses.
    """
    from src.eval import judge

    flag = {r["poem_id"]: r["memorised"] for r in records}
    groups: dict[str, list[dict]] = {"memorised": [], "not_memorised": []}
    for score in scores:
        if score["poem_id"] in flag:
            key = "memorised" if flag[score["poem_id"]] else "not_memorised"
            groups[key].append(score)

    return {name: judge.gaps(rows) if rows else {"n_pairs": 0}
            for name, rows in groups.items()}
