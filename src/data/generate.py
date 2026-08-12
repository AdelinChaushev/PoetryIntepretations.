"""Generate interpretations with the teacher model.

These are the training targets. They do not exist anywhere and cannot be
hand-written at this scale, so a stronger model writes them — which is ordinary
sequence-level distillation, and carries the failure mode the whole project is
built to detect.

**Resumability is not optional.** This is roughly 2,500 API calls, and it will
not complete in one uninterrupted run. Every response is appended to JSONL the
moment it arrives, already-processed poems are skipped on restart, and a
failure is logged rather than raised. A crash costs only the poems still
outstanding.

**One fixed prompt, never changed mid-run.** It lives in
``config.TEACHER_PROMPT_TEMPLATE``. Changing it partway through would make the
corpus a mixture of two tasks and confound every later comparison.
"""

from __future__ import annotations

import json
import logging
import time

import config

log = logging.getLogger(__name__)

_CLIENT = None


def get_client():
    """Return the teacher API client, created on first use.

    DeepSeek exposes an OpenAI-compatible API, so the official OpenAI SDK is
    used with a different ``base_url``. Imported lazily so the module can be
    inspected — and its tests run — without credentials present.
    """
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        _CLIENT = OpenAI(
            api_key=config.require_api_key(config.TEACHER_API_KEY_ENV),
            base_url=config.TEACHER_BASE_URL,
        )
    return _CLIENT


def build_prompt(poem: dict) -> str:
    """Render the fixed template for one poem."""
    return config.TEACHER_PROMPT_TEMPLATE.format(
        title=poem["title"],
        author=poem["author"],
        poem="\n".join(poem["lines"]),
        min_words=config.MIN_WORDS,
        max_words=config.MAX_WORDS,
    )


def interpret(poem: dict) -> str:
    """Ask the teacher for one interpretation. Raises on failure."""
    response = get_client().chat.completions.create(
        model=config.TEACHER_MODEL,
        messages=[{"role": "user", "content": build_prompt(poem)}],
        temperature=config.TEACHER_TEMPERATURE,
        max_tokens=config.TEACHER_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def load_cached() -> list[dict]:
    """Return interpretations already generated, or an empty list."""
    path = config.INTERPRETATIONS_PATH
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def processed_ids() -> set[str]:
    """Poem ids already attempted, so a restart skips them."""
    return {record["poem_id"] for record in load_cached()}


def _append(record: dict) -> None:
    """Append one record immediately — this is what makes restarts cheap."""
    config.INTERPRETATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.INTERPRETATIONS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_or_generate(poems: list[dict], limit: int | None = None) -> list[dict]:
    """Generate interpretations for ``poems``, skipping any already done.

    Args:
        poems: poems to interpret. Filter to the line bounds *before* calling —
            an interpretation of a 16,000-line poem is a wasted API call.
        limit: stop after this many *new* calls. Defaults to
            :data:`config.N_POEMS` under ``SMOKE``, otherwise no limit.

    Failures are logged and skipped, never raised. Losing one poem to a
    timeout must not end a run that has already spent an hour.
    """
    if limit is None and config.SMOKE:
        limit = config.N_POEMS

    # Fail fast on anything that cannot be fixed by trying the next poem. A
    # missing API key is not a per-poem failure — it would fail identically for
    # all 3,156 of them, burning through the whole corpus to report the same
    # error 3,156 times. Credentials are checked once, here, before any work.
    get_client()

    # Poems outside the line bounds are dropped by the funnel later, so
    # interpreting them is a paid API call for output that gets discarded.
    # Filtering here rather than trusting the caller, because the cost of
    # forgetting is real money and an hour of wall-clock.
    from src.data.filter import within_length_bounds

    in_bounds = [poem for poem in poems if within_length_bounds(poem)]
    if len(in_bounds) < len(poems):
        log.info("skipping %d poems outside the length bounds — they would be "
                 "dropped by the funnel anyway",
                 len(poems) - len(in_bounds))

    done = processed_ids()
    if done:
        log.info("resuming: %d interpretations already generated", len(done))

    pending = [poem for poem in in_bounds if poem["poem_id"] not in done]
    if limit is not None:
        pending = pending[:limit]

    log.info("generating %d interpretations with %s",
             len(pending), config.TEACHER_MODEL)

    failures = 0
    for index, poem in enumerate(pending, start=1):
        try:
            text = interpret(poem)
        except Exception as error:  # noqa: BLE001 - one bad call must not stop the run
            failures += 1
            log.error("[%d/%d] %s failed: %s",
                      index, len(pending), poem["poem_id"], error)

            # Consecutive failures mean something systemic — an expired key,
            # a billing limit, an outage — not bad luck on individual poems.
            # Stopping preserves what has been generated and saves the rest of
            # the budget; the run is resumable, so nothing is lost by it.
            if failures >= config.GENERATE_MAX_CONSECUTIVE_FAILURES and \
                    failures == index:
                raise RuntimeError(
                    f"aborting: the first {failures} calls all failed, which "
                    f"points at a systemic problem rather than individual "
                    f"poems. Fix it and re-run — progress so far is cached."
                ) from error
            continue

        _append({
            "poem_id": poem["poem_id"],
            "title": poem["title"],
            "author": poem["author"],
            "interpretation": text,
            "teacher_model": config.TEACHER_MODEL,
        })

        if index % 25 == 0 or index == len(pending):
            log.info("[%d/%d] generated", index, len(pending))
        time.sleep(config.TEACHER_DELAY_SECONDS)

    if failures:
        log.warning("%d poems failed and were skipped; re-run to retry them",
                    failures)

    return load_cached()


def show_example(record: dict) -> None:
    """Print one interpretation readably, for display in a notebook."""
    print(f"{record['title']} — {record['author']}")
    print("-" * 48)
    print(record["interpretation"])
