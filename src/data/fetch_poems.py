"""Fetch poems from PoetryDB and cache them to disk.

PoetryDB is a fixed anthology of 129 authors behind a free HTTP API with no
key. Two endpoints are used: the author list, then poems per author.

Two rules shape this module.

**Cache on arrival, never re-fetch.** Every author's response is appended to
the cache as soon as it returns. A crash halfway through costs only the time
already spent, and re-running skips everything already stored.

**Fetch unfiltered.** Nothing here drops a poem for being too long or too
short — that is :mod:`src.data.filter`'s job. Filtering at fetch time would
mean that changing a threshold later required hitting the API again, which is
exactly what the caching rule exists to prevent.
"""

from __future__ import annotations

import collections
import json
import logging
import re
import time
import unicodedata
from typing import Iterator
from urllib.parse import quote
from urllib.request import Request, urlopen

import config
from src.eval import grounding

log = logging.getLogger(__name__)

#: A leading editorial line number: ``"4 All skillful in the wars;"``. Requires
#: whitespace then a non-space after the digits, so a line that merely opens
#: with a numeral is not matched on that basis alone.
_LINE_NUMBER = re.compile(r"^\s*(\d+)\s+(?=\S)")

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_WORD = re.compile(r"\S+")


#: Reuses the checker's table so the corpus is repaired with exactly the
#: mapping the grounding check folds by — two definitions could drift apart
#: and produce a poem the checker no longer matches its own quotes against.
_MOJIBAKE = grounding.MOJIBAKE_TO_LATIN


def fix_mojibake(poem: dict) -> dict:
    """Repair Latin-1 accents that were decoded as CP1251 Cyrillic.

    Applied per word, and only to words that mix Cyrillic with Latin letters.
    That mixture is the corruption signature — no real word does it — and
    restricting to it means a poem quoting genuine Cyrillic is left untouched
    rather than silently rewritten.

    Worth repairing rather than ignoring: the mangled characters are fed to the
    model as training text, and they fall on the archaic ``-èd`` ending and on
    Hopkins's metrical stress marks, so they cluster in exactly the lines a
    reader would quote.
    """
    def repair(word: str) -> str:
        if not (_CYRILLIC.search(word) and re.search(r"[A-Za-z]", word)):
            return word
        if not all(char in _MOJIBAKE for char in word if _CYRILLIC.match(char)):
            return word
        return "".join(_MOJIBAKE.get(char, char) for char in word)

    lines = [_WORD.sub(lambda m: repair(m.group()), line)
             for line in poem["lines"]]
    return poem if lines == poem["lines"] else {**poem, "lines": lines}


def is_line_numbered(lines: list[str]) -> bool:
    """Whether ``lines`` carry editorial line numbers rather than poem text.

    Two conditions, both required. Enough lines must carry a leading number
    (:data:`config.NUMBERED_LINE_THRESHOLD`), and those numbers must ascend.
    The ascent is what distinguishes an editorial apparatus from a poem that
    happens to use numerals — real numerals do not count upward line by line.

    Numbers are not compared against their position in the list, because blank
    stanza-break lines are unnumbered and make the two drift apart: one poem
    here numbers 40 of its 42 lines while matching its own index on only 12.
    """
    present = [line for line in lines if line.strip()]
    if len(present) < 2:
        return False

    numbers = [int(match.group(1)) for line in present
               if (match := _LINE_NUMBER.match(line))]
    if len(numbers) < 2:
        return False
    if len(numbers) / len(present) < config.NUMBERED_LINE_THRESHOLD:
        return False
    return all(a < b for a, b in zip(numbers, numbers[1:]))


def strip_line_numbers(poem: dict) -> dict:
    """Remove editorial line numbers, leaving the poem otherwise untouched.

    These corrupt two things at once. The number is fed to the model as part of
    the poem, and it sits *between* consecutive lines in the joined text, so a
    correctly-quoted couplet fails the grounding check — a false hallucination
    verdict on a quote that is verbatim right.

    Applied at load time rather than at fetch time, so the cached JSON stays a
    faithful copy of what PoetryDB served. Cleaning is re-derived on every load
    and a change to the rule costs nothing to apply.
    """
    if not is_line_numbered(poem["lines"]):
        return poem
    return {**poem, "lines": [_LINE_NUMBER.sub("", line)
                              for line in poem["lines"]]}


def sort_key(poem: dict) -> tuple[str, str, str]:
    """Ordering used to number poems. Depends only on content.

    Fetch order is not stable — it depends on which authors were already
    cached, and on whether the bulk endpoint or the per-title fallback served
    a given author. Numbering by fetch order would mean that deleting the
    cache and re-fetching renumbers the corpus, silently repointing every
    stored interpretation and fold assignment at a different poem.

    Sorting by content instead makes the numbering reproducible: the same
    corpus yields the same numbers on any machine, in any fetch order.
    """
    return (poem["author"], poem["title"], "\n".join(poem["lines"]))


def assign_ids(poems: list[dict]) -> list[dict]:
    """Number poems 1..N in a deterministic order.

    Sequential integers rather than a hash: they are readable, and the earlier
    hash of author-and-title was not unique — poets reuse titles, so Blake's
    two "Holy Thursday" collided onto one id and an interpretation could be
    scored against a poem the teacher never saw.

    The numbering is positional, so it is only stable while the corpus is
    fixed. Adding or removing a poem shifts every id after it, which would
    invalidate the interpretations and the fold assignment keyed to them. The
    corpus is complete, so that is acceptable — but do not renumber after
    generation has run.
    """
    return [{**poem, "poem_id": index}
            for index, poem in enumerate(sorted(poems, key=sort_key), start=1)]


def _get_json(url: str, max_retries: int | None = None,
              timeout: float | None = None) -> object:
    """GET ``url`` and parse JSON, retrying with exponential backoff.

    PoetryDB errors intermittently, so retries are the expected path rather
    than an exceptional one, and a failure only raises once every attempt is
    used.

    ``max_retries`` is lowered by callers that have a fallback available.
    Retrying hard is right when there is no alternative; when there is one,
    burning 31 seconds of backoff on a failure that is already known to be
    permanent just delays the recovery.
    """
    if max_retries is None:
        max_retries = config.FETCH_MAX_RETRIES
    if timeout is None:
        timeout = config.FETCH_TIMEOUT_SECONDS

    # An explicit User-Agent is required: PoetryDB returns 403 to urllib's
    # default `Python-urllib/x.y`.
    request = Request(url, headers={"User-Agent": config.FETCH_USER_AGENT})

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:  # network, timeout, or malformed JSON
            last_error = error
            if attempt + 1 == max_retries:
                break
            backoff = config.FETCH_BACKOFF_SECONDS * (2**attempt)
            log.warning("attempt %d/%d for %s failed (%s); retrying in %.1fs",
                        attempt + 1, max_retries, url, error, backoff)
            time.sleep(backoff)
    raise RuntimeError(
        f"giving up on {url} after {max_retries} attempts") from last_error


def fetch_author_list() -> list[str]:
    """Return every author name PoetryDB knows about."""
    payload = _get_json(f"{config.POETRYDB_BASE_URL}/author")
    return list(payload["authors"])


def _normalise(record: dict, author: str) -> dict:
    """Convert a PoetryDB record to the project's schema.

    PoetryDB returns ``linecount`` as a *string*; it becomes an ``int`` here so
    nothing downstream has to remember that.
    """
    return {
        "title": record["title"],
        "author": author,
        "lines": record["lines"],
        "linecount": int(record["linecount"]),
    }


def fetch_author_titles(author: str) -> list[str]:
    """Return just the titles by ``author`` — a far lighter response."""
    url = f"{config.POETRYDB_BASE_URL}/author/{quote(author)}/title"
    payload = _get_json(url)
    if isinstance(payload, dict):
        return []
    return [record["title"] for record in payload]


def fetch_poem(author: str, title: str) -> dict | None:
    """Return one poem by author and title, or None if it cannot be fetched."""
    url = (f"{config.POETRYDB_BASE_URL}/author,title/"
           f"{quote(author)};{quote(title)}")
    try:
        # Few retries, short timeout. A single poem that repeatedly times out
        # is one whose response is too large to serve — which means a very long
        # poem, which the length filter discards anyway. There
        # is nothing to be gained by spending 165s to retrieve something that
        # is about to be thrown away.
        payload = _get_json(url,
                            max_retries=config.FETCH_TITLE_RETRIES,
                            timeout=config.FETCH_BULK_TIMEOUT_SECONDS)
    except RuntimeError:
        log.info("skipping %r by %r: too large to serve (would be dropped by "
                 "the line filter regardless)", title, author)
        return None
    if isinstance(payload, dict) or not payload:
        return None
    return _normalise(payload[0], author)


def _fetch_author_poems_individually(author: str) -> list[dict]:
    """Fetch an author one poem at a time, via the title list.

    The fallback for prolific authors: PoetryDB times out building the bulk
    response for the largest collections (Byron and Shelley both 503 after
    ~16s, deterministically — retrying does not help). Titles come back fine,
    so the poems are fetched one at a time instead.
    """
    titles = fetch_author_titles(author)
    log.info("  falling back to per-title fetch for %r (%d titles)",
             author, len(titles))

    poems = []
    for title in titles:
        poem = fetch_poem(author, title)
        if poem is not None:
            poems.append(poem)
        time.sleep(config.FETCH_DELAY_SECONDS)
    return poems


def fetch_author_poems(author: str) -> list[dict]:
    """Return all poems by ``author``, normalised to the project's schema.

    Tries the bulk endpoint first and falls back to per-title fetching if it
    fails, so a slow-to-serve author is not silently dropped from the corpus.
    """
    url = f"{config.POETRYDB_BASE_URL}/author/{quote(author)}"
    try:
        # Two attempts and a short timeout: the per-title fallback below is a
        # better answer than a third try. The server returns its own 503 at
        # ~15s for oversized collections, so the timeout only matters if it
        # hangs instead of erroring — and then waiting is pure loss.
        payload = _get_json(url,
                            max_retries=config.FETCH_BULK_RETRIES,
                            timeout=config.FETCH_BULK_TIMEOUT_SECONDS)
    except RuntimeError:
        log.info("bulk endpoint will not serve %r (likely too large)", author)
        return _fetch_author_poems_individually(author)

    if isinstance(payload, dict):  # PoetryDB signals "not found" as an object
        log.warning("no poems for %r: %s", author, payload)
        return []

    return [_normalise(record, author) for record in payload]


def _deduplicate(poems: list[dict]) -> list[dict]:
    """Drop records that repeat a poem already seen, keyed on content.

    PoetryDB returns some poems twice, occasionally with different
    transcription (``Sit still a word`` vs ``SIT stilla word``). Keying on
    author, title and text means a genuine repeat collapses while two distinct
    poems that merely share a title — Blake's two "Holy Thursday", Poe's two
    "To Helen" — are both kept, as they should be.
    """
    seen, kept = set(), []
    for poem in poems:
        key = sort_key(poem)
        if key not in seen:
            seen.add(key)
            kept.append(poem)

    if len(kept) < len(poems):
        log.info("dropped %d duplicate records", len(poems) - len(kept))
    return kept


def load_cached() -> list[dict]:
    """Return every cached poem, cleaned, deduplicated and numbered 1..N.

    Line-number stripping runs *before* numbering, so ids are assigned from the
    cleaned text and stay consistent with what the rest of the pipeline sees.
    It affects a handful of poems and leaves ``sort_key`` unchanged for all of
    them — text is only the third sort field, behind author and title — so ids
    do not move. ``tests/test_fetch_poems.py`` pins that.
    """
    path = config.RAW_POEMS_PATH
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    cleaned = [strip_line_numbers(fix_mojibake(record))
               for record in records]
    return assign_ids(_deduplicate(cleaned))


def cached_authors() -> set[str]:
    """Return the authors already fetched, so they can be skipped on restart."""
    return {poem["author"] for poem in load_cached()}


def _append(poems: list[dict]) -> None:
    """Append poems to the cache immediately, one JSON object per line."""
    config.RAW_POEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.RAW_POEMS_PATH.open("a", encoding="utf-8") as handle:
        for poem in poems:
            handle.write(json.dumps(poem, ensure_ascii=False) + "\n")


def load_or_fetch(max_authors: int | None = None) -> list[dict]:
    """Return the full raw corpus, fetching only what is not already cached.

    Resumable by construction: authors already present in the cache are
    skipped, and each author's poems are written before the next request is
    made. Individual author failures are logged and skipped rather than
    aborting a run that may already have taken many minutes.

    Args:
        max_authors: stop after this many *new* authors. Defaults to
            :data:`config.SMOKE_MAX_AUTHORS` under ``SMOKE``, otherwise no
            limit — the whole anthology is small enough to take in full.
    """
    if max_authors is None and config.SMOKE:
        max_authors = config.SMOKE_MAX_AUTHORS

    already_done = cached_authors()
    if already_done:
        log.info("resuming: %d authors already cached", len(already_done))

    pending = [a for a in fetch_author_list() if a not in already_done]
    if max_authors is not None:
        pending = pending[:max_authors]

    failures = 0
    for index, author in enumerate(pending, start=1):
        try:
            poems = fetch_author_poems(author)
        except RuntimeError as error:
            failures += 1
            log.error("skipping %r: %s", author, error)
            continue

        _append(poems)
        log.info("[%d/%d] %s: %d poems", index, len(pending), author, len(poems))
        time.sleep(config.FETCH_DELAY_SECONDS)

    if failures:
        log.warning("%d authors failed and were skipped", failures)

    return load_cached()


def iter_poems() -> Iterator[dict]:
    """Stream cached poems without holding the whole corpus in memory."""
    with config.RAW_POEMS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def describe_corpus(poems: list[dict]) -> str:
    """Report whether the fetch is complete. Deliberately not a statistics dump.

    This runs immediately after fetching, where the only question is whether
    everything arrived: PoetryDB 403s an unset user agent and 503s the largest
    per-author collections, so a partial fetch is the expected failure and it
    is silent — the pipeline downstream would simply run on a smaller corpus.

    Length distributions, author skew and the singleton count are **not** here.
    They describe the corpus that survives filtering rather than the one that
    was downloaded, they are plotted in Figure 2 where a shape is readable in a
    way a median is not, and printing them twice invites the two to disagree.
    What was dropped, and why, belongs to the funnel in :mod:`src.data.filter`.
    """
    authors = collections.Counter(poem["author"] for poem in poems)
    thin = [name for name, count in sorted(authors.items()) if count == 1]

    lines = [
        f"fetched            {len(poems)} poems from {len(authors)} authors",
    ]
    if config.POETRYDB_AUTHOR_COUNT:
        missing = config.POETRYDB_AUTHOR_COUNT - len(authors)
        lines.append(
            f"author coverage    {len(authors)}/{config.POETRYDB_AUTHOR_COUNT}"
            + (f"  — {missing} MISSING, re-run the fetch" if missing > 0 else "  complete")
        )
    if thin:
        lines.append(f"single-poem authors {len(thin)}  "
                     f"(a truncated fetch looks like this)")
    return "\n".join(lines)


def show_example(poem: dict, max_lines: int | None = 8) -> None:
    """Print one poem readably, for display in a notebook.

    ``max_lines=None`` prints the poem in full. Use that wherever the poem is
    being read against an interpretation — a truncated poem cannot be checked
    against a quote drawn from the part that was cut.
    """
    print(f"{poem['title']}\n{poem['author']}  ({poem['linecount']} lines)")
    print("-" * 60)

    lines = poem["lines"] if max_lines is None else poem["lines"][:max_lines]
    for line in lines:
        print(line)

    remaining = poem["linecount"] - len(lines)
    if remaining > 0:
        print(f"... ({remaining} more lines)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    corpus = load_or_fetch()
    authors = {poem["author"] for poem in corpus}
    print(f"\n{len(corpus)} poems from {len(authors)} authors "
          f"cached at {config.RAW_POEMS_PATH}")
