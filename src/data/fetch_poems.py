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
import hashlib
import json
import logging
import re
import time
from typing import Iterator
from urllib.parse import quote
from urllib.request import Request, urlopen

import config

log = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, max_length: int = 30) -> str:
    """Lowercase, hyphen-separated form of ``text``, truncated."""
    return _SLUG_RE.sub("-", text.lower()).strip("-")[:max_length]


def make_poem_id(author: str, title: str) -> str:
    """Return a stable, readable id for a poem.

    Deterministic across machines and runs, because the fold assignment is
    built from these ids locally and shipped to Kaggle — an id that changed
    between environments would silently break the held-out guarantee.

    The trailing hash disambiguates poems whose titles collide after slugging.
    """
    digest = hashlib.sha1(f"{author}|{title}".encode()).hexdigest()[:6]
    return f"{_slug(author, 20)}--{_slug(title)}--{digest}"


def _get_json(url: str) -> object:
    """GET ``url`` and parse JSON, retrying with exponential backoff.

    PoetryDB errors intermittently. Retries are the expected path, not an
    exceptional one, so a failure only raises after every attempt is used.
    """
    # An explicit User-Agent is required: PoetryDB returns 403 to urllib's
    # default `Python-urllib/x.y`.
    request = Request(url, headers={"User-Agent": config.FETCH_USER_AGENT})

    last_error: Exception | None = None
    for attempt in range(config.FETCH_MAX_RETRIES):
        try:
            with urlopen(request, timeout=config.FETCH_TIMEOUT_SECONDS) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:  # network, timeout, or malformed JSON
            last_error = error
            backoff = config.FETCH_BACKOFF_SECONDS * (2**attempt)
            log.warning("attempt %d for %s failed (%s); retrying in %.1fs",
                        attempt + 1, url, error, backoff)
            time.sleep(backoff)
    raise RuntimeError(f"giving up on {url} after "
                       f"{config.FETCH_MAX_RETRIES} attempts") from last_error


def fetch_author_list() -> list[str]:
    """Return every author name PoetryDB knows about."""
    payload = _get_json(f"{config.POETRYDB_BASE_URL}/author")
    return list(payload["authors"])


def _normalise(record: dict, author: str) -> dict:
    """Convert a PoetryDB record to the project's schema.

    PoetryDB returns ``linecount`` as a *string*; it becomes an ``int`` here so
    nothing downstream has to remember that.
    """
    title = record["title"]
    return {
        "poem_id": make_poem_id(author, title),
        "title": title,
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
        payload = _get_json(url)
    except RuntimeError as error:
        log.warning("could not fetch %r by %r: %s", title, author, error)
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
        payload = _get_json(url)
    except RuntimeError as error:
        log.warning("bulk fetch failed for %r (%s)", author, error)
        return _fetch_author_poems_individually(author)

    if isinstance(payload, dict):  # PoetryDB signals "not found" as an object
        log.warning("no poems for %r: %s", author, payload)
        return []

    return [_normalise(record, author) for record in payload]


def load_cached() -> list[dict]:
    """Return every poem already cached, or an empty list if there is none."""
    path = config.RAW_POEMS_PATH
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    """Return a summary of the raw corpus, before any filtering.

    Reported in the data notebook so the corpus is characterised by what was
    actually retrieved rather than by the number that was hoped for. The
    in-range count here is a *preview* of one filter stage — the authoritative
    funnel is built in :mod:`src.data.filter`.
    """
    authors = collections.Counter(poem["author"] for poem in poems)
    line_counts = sorted(poem["linecount"] for poem in poems)
    in_range = [
        poem for poem in poems
        if config.MIN_LINES <= poem["linecount"] <= config.MAX_LINES
    ]
    with_sibling = sum(
        1 for count in collections.Counter(
            poem["author"] for poem in in_range
        ).values() if count >= config.MIN_POEMS_PER_AUTHOR_FOR_EVAL
    )

    biggest = authors.most_common(5)
    width = max(len(name) for name, _ in biggest)
    share = "\n".join(
        f"    {name:<{width}}  {count:>4}  ({count / len(poems):.1%})"
        for name, count in biggest
    )

    return (
        f"raw corpus         {len(poems)} poems from {len(authors)} authors\n"
        f"lines              median {line_counts[len(line_counts) // 2]}, "
        f"min {line_counts[0]}, max {line_counts[-1]}\n"
        f"within [{config.MIN_LINES}, {config.MAX_LINES}] lines   "
        f"{len(in_range)} ({len(in_range) / len(poems):.0%})\n"
        f"authors with >= {config.MIN_POEMS_PER_AUTHOR_FOR_EVAL} in-range poems   "
        f"{with_sibling}\n"
        f"\n  largest collections — these dominate the grouped folds:\n{share}"
    )


def show_example(poem: dict, max_lines: int = 8) -> None:
    """Print one poem readably, for display in a notebook."""
    print(f"{poem['title']}\n{poem['author']}  ({poem['linecount']} lines)")
    print("-" * 48)
    for line in poem["lines"][:max_lines]:
        print(line)
    if poem["linecount"] > max_lines:
        print(f"... ({poem['linecount'] - max_lines} more lines)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    corpus = load_or_fetch()
    authors = {poem["author"] for poem in corpus}
    print(f"\n{len(corpus)} poems from {len(authors)} authors "
          f"cached at {config.RAW_POEMS_PATH}")
