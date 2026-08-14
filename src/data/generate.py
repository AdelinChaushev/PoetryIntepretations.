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


def interpret_until_grounded(poem: dict) -> dict:
    """Interpret ``poem``, resampling while the teacher misquotes it.

    A hallucinated quote is a bad draw from the sampler, not a property of the
    poem — the same class of failure as a timeout, which is already retried.
    Dropping the poem instead would discard usable input over a recoverable
    output failure, and would not be neutral: the drop falls hardest on long
    poems and on Byron, biasing the corpus toward what the teacher found easy.

    **The first attempt's verdict is recorded and never overwritten.** The
    reported teacher hallucination rate is a finding about the teacher, so it
    must come from first attempts; resampling exists to save the poem, not to
    improve the number. Reading the rate off the final corpus instead would
    report a teacher that does not exist.
    """
    from src.eval import grounding

    first_grounded = None
    text = ""

    for attempt in range(1, config.GENERATE_MAX_ATTEMPTS + 1):
        text = interpret(poem)
        grounded = grounding.check(text, poem)["grounded"]
        if first_grounded is None:
            first_grounded = grounded
        if grounded:
            break
        if attempt < config.GENERATE_MAX_ATTEMPTS:
            time.sleep(config.TEACHER_DELAY_SECONDS)

    return {
        "poem_id": poem["poem_id"],
        "title": poem["title"],
        "author": poem["author"],
        "interpretation": text,
        "teacher_model": config.TEACHER_MODEL,
        "attempts": attempt,
        "grounded": grounded,
        "first_attempt_grounded": first_grounded,
    }


def load_cached() -> list[dict]:
    """Return the current interpretation for each poem.

    The cache is append-only, so a resampled poem has more than one record.
    The newest wins — but two fields are accumulated across records rather than
    overwritten, because both are measurements of the teacher that a later
    success must not erase:

    ``first_attempt_grounded`` is carried forward from the oldest record.

    ``total_attempts`` sums ``attempts`` across every record. The per-record
    ``attempts`` counts only the calls made *within one run*, so a poem that
    failed once, was picked up by a later run and succeeded immediately stores
    ``attempts=1`` while having genuinely cost two calls. Reading the stored
    field directly would report resampling as rarer than it was.
    """
    path = config.INTERPRETATIONS_PATH
    if not path.exists():
        return []

    by_id: dict = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            attempts = record.get("attempts", 1)

            previous = by_id.get(record["poem_id"])
            if previous is None:
                record = {**record, "total_attempts": attempts}
            else:
                carried = {"total_attempts": previous["total_attempts"] + attempts}
                if "first_attempt_grounded" in previous:
                    carried["first_attempt_grounded"] = \
                        previous["first_attempt_grounded"]
                record = {**record, **carried}

            by_id[record["poem_id"]] = record

    return list(by_id.values())


def first_attempts() -> list[dict]:
    """Return the **oldest** record for each poem — the teacher's first draw.

    The mirror of :func:`load_cached`, which returns the newest. The cache is
    append-only, so the text of every first attempt survives resampling even
    though the corpus no longer uses it. That matters more than it sounds:
    ``first_attempt_grounded`` was written by whatever version of the grounding
    checker was current at generation time, so a later fix to the checker
    cannot reach it. Recomputing from the preserved text can.

    This is what makes the reported teacher hallucination rate reproducible.
    Scoring these records with today's checker gives the rate today's checker
    implies, on any corpus, rather than a number frozen at the moment of an
    API call — which is why notebook 01 measures rather than quotes it.
    """
    path = config.INTERPRETATIONS_PATH
    if not path.exists():
        return []

    oldest: dict = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            oldest.setdefault(record["poem_id"], record)

    return list(oldest.values())


def processed_ids(poems: list[dict] | None = None,
                  retry_ungrounded: bool = False) -> set[str]:
    """Poem ids that need no further work.

    A poem is done when its interpretation is grounded, or when the attempt
    budget is spent. An ungrounded interpretation with attempts left is *not*
    done: the poem is fine and the sampler simply drew badly, so it is
    resampled rather than dropped from the corpus.

    Args:
        poems: judge groundedness with the **current** checker instead of the
            stored ``grounded`` flag. The flag is written at generation time,
            so a poem rejected by a checker that was later fixed stays marked
            ungrounded forever, and re-running would spend API calls
            re-interpreting text that is already correct.
        retry_ungrounded: give still-ungrounded poems another
            :data:`config.GENERATE_MAX_ATTEMPTS` calls, ignoring the per-run
            budget they already spent. ``attempts`` counts calls within one
            run, so without this a poem that exhausted a run is skipped by
            every future run — permanently, with no way back. Bounded by
            :data:`config.GENERATE_MAX_TOTAL_ATTEMPTS` so a poem the teacher
            simply cannot quote does not absorb calls forever.
    """
    by_id = {poem["poem_id"]: poem for poem in poems} if poems else {}
    done = set()

    for record in load_cached():
        poem = by_id.get(record["poem_id"]) if by_id else None
        if poem is not None:
            from src.eval import grounding
            grounded = grounding.check(record["interpretation"], poem)["grounded"]
        else:
            grounded = record.get("grounded")

        attempts = record.get("attempts", 1)
        total = record.get("total_attempts", attempts)

        if grounded is None or grounded:
            done.add(record["poem_id"])
        elif retry_ungrounded:
            if total >= config.GENERATE_MAX_TOTAL_ATTEMPTS:
                done.add(record["poem_id"])
        elif attempts >= config.GENERATE_MAX_ATTEMPTS:
            done.add(record["poem_id"])

    return done


def _append(record: dict) -> None:
    """Append one record immediately — this is what makes restarts cheap."""
    config.INTERPRETATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.INTERPRETATIONS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _quieten_http_logs() -> None:
    """Quieten the HTTP client for the duration of a run.

    Delegates to :func:`config.configure_logging` so the list of noisy loggers
    lives in one place — a second copy here would drift, and the failure mode
    is a run whose real errors are buried under thousands of "200 OK" lines.
    """
    config.configure_logging()


def _progress(pending: list[dict], cached: int):
    """Return a progress bar over ``pending``, or None if tqdm is unavailable.

    Shows generated/total as a single updating line rather than one log line
    per poem — a 2,600-call run should look like progress, not a wall of text.
    """
    if not pending:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - optional convenience only
        return None

    return tqdm(total=len(pending), unit="poem",
                desc=f"{config.TEACHER_MODEL} ({cached} cached)"
                if cached else config.TEACHER_MODEL)


def _write(progress, message: str) -> None:
    """Emit a message without corrupting the progress bar's line."""
    if progress is None:
        log.error(message)
    else:
        progress.write(message)


def _report_withheld(poems: list[dict], done: set, retry_ungrounded: bool) -> None:
    """Log poems that are still ungrounded but will not be retried, and why.

    "nothing to generate" is otherwise indistinguishable from "everything
    succeeded", which is the wrong thing to tell someone who has just asked for
    the failures to be redone. Both reasons for holding a poem back are silent
    by construction — one is a default argument, the other a ceiling in
    ``config`` — so neither shows up anywhere unless it is said out loud.
    """
    from src.eval import grounding

    cached = {record["poem_id"]: record for record in load_cached()}
    stuck = []
    for poem in poems:
        record = cached.get(poem["poem_id"])
        if record is None or poem["poem_id"] not in done:
            continue
        if not grounding.check(record["interpretation"], poem)["grounded"]:
            stuck.append(record)

    if not stuck:
        return

    if not retry_ungrounded:
        log.warning("%d poems are still ungrounded and were NOT retried; "
                    "pass retry_ungrounded=True to try them again", len(stuck))
        return

    ceiling = sum(1 for record in stuck
                  if record.get("total_attempts", record.get("attempts", 1))
                  >= config.GENERATE_MAX_TOTAL_ATTEMPTS)
    if ceiling:
        log.warning("%d poems are still ungrounded after "
                    "GENERATE_MAX_TOTAL_ATTEMPTS=%d attempts and will not be "
                    "retried again — this is the residue the teacher cannot "
                    "reliably quote, and it belongs in the writeup rather than "
                    "in another run", ceiling, config.GENERATE_MAX_TOTAL_ATTEMPTS)
    if len(stuck) > ceiling:
        log.warning("%d further poems are ungrounded but below the ceiling",
                    len(stuck) - ceiling)


def load_or_generate(poems: list[dict], limit: int | None = None,
                     retry_ungrounded: bool = False) -> list[dict]:
    """Generate interpretations for ``poems``, skipping any already done.

    Args:
        poems: poems to interpret. Filter to the line bounds *before* calling —
            an interpretation of a 16,000-line poem is a wasted API call.
        limit: how many interpretations should **exist** for ``poems`` when
            this returns — a target, not a count of new API calls. If that many
            are already cached, nothing is generated and the call is free.
            Defaults to :data:`config.N_POEMS` under ``SMOKE``, otherwise no
            limit.
        retry_ungrounded: re-queue poems whose stored interpretation still
            misquotes, giving them a further :data:`config.GENERATE_MAX_ATTEMPTS`
            calls up to :data:`config.GENERATE_MAX_TOTAL_ATTEMPTS` overall.
            Off by default because it costs API calls; the reported teacher
            hallucination rate is unaffected either way, since it is measured
            on first attempts.

    Returns:
        Interpretations for ``poems`` only, capped at ``limit`` — not the whole
        cache. A pilot asking for 30 gets the same 30 every time.

    Failures are logged and skipped, never raised. Losing one poem to a
    timeout must not end a run that has already spent an hour.
    """
    if limit is None and config.SMOKE:
        limit = config.N_POEMS

    # Fail fast on anything that cannot be fixed by trying the next poem. A
    # missing API key is not a per-poem failure — it would fail identically for
    # every poem, burning through the whole corpus to report the same error
    # thousands of times.
    #
    # This reads the environment directly rather than calling get_client(),
    # which caches its client in a module global: once that cache is warm the
    # credential check never runs again, so routing this through it would make
    # the guard silently no-op in exactly the long-lived process — a notebook
    # kernel — where it matters most.
    config.require_api_key(config.TEACHER_API_KEY_ENV)

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

    done = processed_ids(poems=in_bounds, retry_ungrounded=retry_ungrounded)
    cached = [poem for poem in in_bounds if poem["poem_id"] in done]
    pending = [poem for poem in in_bounds if poem["poem_id"] not in done]

    # `limit` is a target, not a batch size. Taking only the shortfall is what
    # makes re-running free: a pilot cell someone re-runs while reading the
    # output must not quietly buy another 30 interpretations each time.
    if limit is not None:
        cached = cached[:limit]
        pending = pending[:max(0, limit - len(cached))]

    if cached:
        log.info("reusing %d cached interpretations", len(cached))

    _report_withheld(in_bounds, done, retry_ungrounded)

    if not pending:
        log.info("nothing to generate")
    else:
        log.info("generating %d interpretations with %s",
                 len(pending), config.TEACHER_MODEL)

    # The HTTP client logs a line per request at INFO. Over 2,600 calls that
    # buries every message that matters — including the failures — under
    # thousands of "200 OK" lines. Quieten it for the duration of the run.
    _quieten_http_logs()

    failures = 0
    progress = _progress(pending, cached=len(cached))

    for index, poem in enumerate(pending, start=1):
        try:
            record = interpret_until_grounded(poem)
        except Exception as error:  # noqa: BLE001 - one bad call must not stop the run
            failures += 1
            _write(progress, f"failed: {poem['poem_id']}: {error}")

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

        _append(record)

        if progress is not None:
            progress.update(1)
        time.sleep(config.TEACHER_DELAY_SECONDS)

    if progress is not None:
        progress.close()

    if failures:
        log.warning("%d poems failed and were skipped; re-run to retry them",
                    failures)

    # Return interpretations for the requested poems only, in the order the
    # poems were given, capped at `limit`. Returning the whole cache instead
    # would mean a pilot's metrics silently widened to cover every poem ever
    # generated, including ones the caller did not ask about.
    wanted = [poem["poem_id"] for poem in (cached + pending)]
    by_id = {record["poem_id"]: record for record in load_cached()}
    return [by_id[pid] for pid in wanted if pid in by_id]


def show_example(record: dict) -> None:
    """Print one interpretation readably, for display in a notebook."""
    print(f"{record['title']} — {record['author']}")
    print("-" * 48)
    print(record["interpretation"])
