"""Score interpretation/poem pairs with an LLM judge.

Two judges from different families, and **they are never pooled**. The primary
(GPT-4o-mini) supplies every hypothesis test and every headline number; the
secondary (Gemini 2.5 Flash) re-runs each one as a robustness check and is
reported alongside. Averaging them would produce a composite nobody can
interpret, and choosing which to report after seeing results is exactly the
researcher freedom the pre-registration exists to remove.

That is enforced rather than documented: every record carries its judge name,
and :func:`assert_single_judge` raises on a mixed set instead of quietly
averaging across two different instruments.

**Resumable, like generation.** This is thousands of paid calls. Every score is
appended to JSONL the moment it arrives, already-scored pairs are skipped on
restart, and a failure is logged rather than raised.

**The scale is never calibrated, only kept consistent.** The swap test is a
difference between conditions scored by the same judge with the same prompt, so
miscalibration moves every condition together and cancels. That is what lets
this work without ground truth.
"""

from __future__ import annotations

import json
import logging
import re
import time

import config

log = logging.getLogger(__name__)

_CLIENTS: dict[str, object] = {}

#: A bare integer, or one embedded in a short reply. The prompt asks for a
#: single integer, but a judge that adds "Score: 8" must not be scored as a
#: failure — and one that writes an essay must not have its first stray number
#: silently taken as the verdict, which is why the match is anchored.
_SCORE = re.compile(r"^\D{0,12}?(\d{1,2})\b")


def scores_path(judge: config.JudgeSpec) -> "object":
    """Where one judge's scores are cached.

    Keyed by judge name in the FILENAME, not only inside the records, so two
    judges cannot write to one file even by mistake.
    """
    return config.RESULTS_DIR / f"judge_scores_{judge.name}.jsonl"


def get_client(judge: config.JudgeSpec):
    """Return an API client for ``judge``, created on first use.

    Both judges are reached through the OpenAI SDK: Gemini exposes an
    OpenAI-compatible endpoint, so one code path serves both and the two cannot
    drift apart in retry behaviour or parameter handling.
    """
    if judge.name not in _CLIENTS:
        from openai import OpenAI

        base_url = judge.base_url or (
            "https://generativelanguage.googleapis.com/v1beta/openai/"
            if "gemini" in judge.model else None
        )
        _CLIENTS[judge.name] = OpenAI(
            api_key=config.require_api_key(judge.api_key_env),
            base_url=base_url,
        )
    return _CLIENTS[judge.name]


def build_prompt(interpretation: str, poem: dict) -> str:
    """Render the fixed swap-test prompt for one pair."""
    return config.SWAP_JUDGE_PROMPT_TEMPLATE.format(
        title=poem["title"],
        author=poem["author"],
        poem="\n".join(poem["lines"]),
        interpretation=interpretation,
        min_score=config.JUDGE_SCORE_MIN,
        max_score=config.JUDGE_SCORE_MAX,
    )


def parse_score(reply: str) -> int | None:
    """Extract the integer score, or None if the reply is unusable.

    Returns None rather than a default. A malformed reply scored as 5 would be
    indistinguishable from a genuine middling verdict, and it would drag the
    gap toward zero — biasing against the project's own hypotheses in a way
    nobody could see in the aggregate.
    """
    match = _SCORE.match(reply.strip())
    if not match:
        return None
    value = int(match.group(1))
    if not config.JUDGE_SCORE_MIN <= value <= config.JUDGE_SCORE_MAX:
        return None
    return value


def _is_transient(error: Exception) -> bool:
    """Whether retrying could plausibly succeed.

    Rate limits and 5xx are worth waiting out. Authentication failures, a
    missing model and a spent quota are not — retrying those burns the backoff
    budget on every pair to arrive at the same answer, and buries the real
    cause under hundreds of identical messages.
    """
    text = str(error).lower()
    if "quota" in text or "billing" in text or "insufficient" in text:
        return False
    status = getattr(error, "status_code", None)
    return status == 429 or (status is not None and 500 <= status < 600)


def score_pair(pair, poem: dict, judge: config.JudgeSpec) -> dict:
    """Score one pair, retrying transient failures with exponential backoff.

    Raises once the retries are spent; returns ``score=None`` on a reply that
    arrived but could not be parsed. Those are different failures and are kept
    apart: the first is the API, the second is the judge.
    """
    prompt = build_prompt(pair.interpretation, poem)
    last_error: Exception | None = None

    for attempt in range(config.JUDGE_MAX_RETRIES):
        try:
            response = get_client(judge).chat.completions.create(
                model=judge.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.JUDGE_TEMPERATURE,
                max_tokens=config.JUDGE_MAX_TOKENS,
            )
            break
        except Exception as error:  # noqa: BLE001 - re-raised below
            last_error = error
            if not _is_transient(error) or attempt + 1 == config.JUDGE_MAX_RETRIES:
                raise
            wait = config.JUDGE_BACKOFF_SECONDS * (2 ** attempt)
            log.debug("%s: %s, retrying in %.1fs", judge.name,
                      type(error).__name__, wait)
            time.sleep(wait)
    else:  # pragma: no cover - loop always breaks or raises
        raise last_error

    reply = (response.choices[0].message.content or "").strip()
    return {
        "judge": judge.name,
        "judge_model": judge.model,
        "arm": pair.arm,
        "poem_id": pair.poem_id,
        "shown_id": pair.shown_id,
        "condition": pair.condition,
        "score": parse_score(reply),
        "reply": reply[:80],
    }


def _key(record) -> tuple:
    """What makes a scored pair unique within one judge's cache."""
    if isinstance(record, dict):
        return (record["arm"], record["poem_id"], record["condition"])
    return (record.arm, record.poem_id, record.condition)


def load_cached(judge: config.JudgeSpec) -> list[dict]:
    """Return this judge's scores, newest record per pair winning."""
    path = scores_path(judge)
    if not path.exists():
        return []

    by_key: dict = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                by_key[_key(record)] = record
    return list(by_key.values())


def _append(judge: config.JudgeSpec, record: dict) -> None:
    path = scores_path(judge)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def score_all(pairs, poems: list[dict], judge: config.JudgeSpec,
              limit: int | None = None) -> list[dict]:
    """Score every pair with ``judge``, skipping any already cached.

    Args:
        pairs: from :func:`src.eval.swap_test.build_pairs`.
        poems: the corpus, so ``shown_id`` can be resolved to a poem.
        judge: which judge. Its name keys both the cache file and every record.
        limit: how many scores should **exist** when this returns — a target,
            not a count of new calls, so re-running a cell costs nothing.

    Failures are logged and skipped, never raised: losing one pair to a timeout
    must not end a run that has already spent an hour and real money.
    """
    config.require_api_key(judge.api_key_env)
    by_id = {poem["poem_id"]: poem for poem in poems}

    done = {_key(record) for record in load_cached(judge)
            if record.get("score") is not None}
    pending = [pair for pair in pairs if _key(pair) not in done]
    if limit is not None:
        pending = pending[:max(0, limit - (len(pairs) - len(pending)))]

    log.info("%s: %d cached, %d to score", judge.name,
             len(pairs) - len(pending), len(pending))

    failures = 0
    progress = _progress(pending, judge)
    for index, pair in enumerate(pending, start=1):
        try:
            record = score_pair(pair, by_id[pair.shown_id], judge)
        except Exception as error:  # noqa: BLE001 - one bad call must not stop the run
            failures += 1
            _write(progress, f"failed {pair.poem_id}/{pair.condition}: {error}")
            # Consecutive failures from the first call mean something systemic —
            # a bad key, a billing limit, an outage — not bad luck on one pair.
            if failures >= config.GENERATE_MAX_CONSECUTIVE_FAILURES and \
                    failures == index:
                raise RuntimeError(
                    f"aborting: the first {failures} {judge.name} calls all "
                    f"failed, which points at a systemic problem. Progress so "
                    f"far is cached."
                ) from error
            continue

        _append(judge, record)
        if progress is not None:
            progress.update(1)
        time.sleep(config.JUDGE_DELAY_SECONDS)

    if progress is not None:
        progress.close()
    if failures:
        log.warning("%s: %d pairs failed and were skipped; re-run to retry",
                    judge.name, failures)

    wanted = {_key(pair) for pair in pairs}
    return [record for record in load_cached(judge) if _key(record) in wanted]


def _progress(pending, judge: config.JudgeSpec):
    if not pending:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - optional convenience only
        return None
    return tqdm(total=len(pending), unit="pair", desc=judge.name)


def _write(progress, message: str) -> None:
    if progress is None:
        log.error(message)
    else:
        progress.write(message)


# --- aggregation ------------------------------------------------------------

def assert_single_judge(records: list[dict]) -> str:
    """Return the one judge these records came from, or raise.

    Pooling is the failure mode that would quietly turn a robustness check into
    a composite metric nobody can interpret. It raises here rather than
    averaging, because a mean over two instruments looks exactly like a mean
    over one.
    """
    names = {record["judge"] for record in records}
    assert len(names) == 1, (
        f"records span {sorted(names)} — judges are never pooled. Filter to one "
        f"judge and report the second alongside, never averaged into it."
    )
    return names.pop()


def condition_means(records: list[dict]) -> dict[str, float]:
    """Mean score per condition, for a single judge."""
    assert_single_judge(records)
    sums: dict[str, list[int]] = {}
    for record in records:
        if record.get("score") is not None:
            sums.setdefault(record["condition"], []).append(record["score"])
    return {condition: sum(scores) / len(scores)
            for condition, scores in sorted(sums.items()) if scores}


def paired_differences(records: list[dict], left: str,
                       right: str) -> list[float]:
    """Per-poem ``left − right`` where both conditions were scored.

    Paired, not two independent means: the same interpretation appears in every
    condition, so pairing removes between-poem variance and is what the
    bootstrap later resamples.
    """
    assert_single_judge(records)
    by_poem: dict[int, dict[str, int]] = {}
    for record in records:
        if record.get("score") is not None:
            by_poem.setdefault(record["poem_id"], {})[record["condition"]] = \
                record["score"]
    return [scores[left] - scores[right]
            for scores in by_poem.values()
            if left in scores and right in scores]


def gaps(records: list[dict]) -> dict[str, float]:
    """The two gaps and the author-level component between them.

    ``grounding_gap``
        matched − mismatched_random. The standard measurement.
    ``poem_level_gap``
        matched − mismatched_same_author. The strict one, and the defensible
        number: it survives author-prior leakage, including leakage that
        arrived during pretraining and that no fold structure can prevent.
    ``author_component``
        the difference between them — how much of the apparent grounding is
        recognising the author rather than reading this poem.
    """
    standard = paired_differences(records, "matched", "mismatched_random")
    strict = paired_differences(records, "matched", "mismatched_same_author")
    mean = lambda values: sum(values) / len(values) if values else float("nan")
    grounding, poem_level = mean(standard), mean(strict)
    return {
        "grounding_gap": grounding,
        "poem_level_gap": poem_level,
        "author_component": grounding - poem_level,
        "n_pairs": len(standard),
    }


def summary_row(records: list[dict]) -> dict:
    """One judge's swap-test result as a flat row.

    Raises via :func:`assert_single_judge` if handed a mixed set, so a summary
    table can never contain a row that is secretly an average of two judges.
    """
    name = assert_single_judge(records)
    usable = [record for record in records if record.get("score") is not None]
    means = condition_means(records)
    measured = gaps(records)

    return {
        "judge": name,
        "judge_model": records[0].get("judge_model", ""),
        "arm": records[0].get("arm", ""),
        "n_pairs": len(records),
        "n_scored": len(usable),
        "n_unparseable": len(records) - len(usable),
        **{f"mean_{condition}": round(value, 3)
           for condition, value in means.items()},
        "grounding_gap": round(measured["grounding_gap"], 3),
        "poem_level_gap": round(measured["poem_level_gap"], 3),
        "author_component": round(measured["author_component"], 3),
        "n_paired_poems": measured["n_pairs"],
    }


def save_summary(records_by_judge: list[list[dict]], path=None):
    """Write one row per judge to CSV and return the table.

    Judges occupy separate ROWS, never a pooled mean. Reporting the primary as
    the result and the secondary beside it is the whole design; a table that
    averaged them would answer a question nobody asked.
    """
    import pandas as pd

    if path is None:
        path = config.SWAP_SUMMARY_CSV_PATH
    table = pd.DataFrame([summary_row(records) for records in records_by_judge
                          if records])
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    log.info("wrote %s", path)
    return table


def separation_verdict(records: list[dict]) -> tuple[bool, str]:
    """Does this judge separate matched from mismatched well enough to use?

    The day-2 gate. If the judge cannot tell a matched interpretation from a
    mismatched one on TEACHER outputs — text known to have been written from
    the poem — then it cannot measure grounding on model outputs either, and
    the evaluation design has to change before any GPU time is spent.

    Reports the standard gap against ``config.MIN_JUDGE_SEPARATION`` rather
    than a p-value: with 150 paired observations almost any non-zero gap is
    significant, and the question here is whether the instrument discriminates
    usefully, not whether it discriminates detectably.
    """
    measured = gaps(records)
    gap = measured["grounding_gap"]
    passed = gap >= config.MIN_JUDGE_SEPARATION
    verdict = (
        f"PASS — matched scores {gap:.2f} above mismatched_random "
        f"(threshold {config.MIN_JUDGE_SEPARATION})"
        if passed else
        f"FAIL — matched scores only {gap:.2f} above mismatched_random, "
        f"below the {config.MIN_JUDGE_SEPARATION} threshold. The judge cannot "
        f"tell grounded from ungrounded on text known to be grounded, so it "
        f"cannot measure the arms either. The design must change."
    )
    return passed, verdict
