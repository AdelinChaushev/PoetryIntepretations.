"""Record what produced a result, so a reader can check rather than trust.

**Exact reproduction is not achievable here, and claiming it would be false.**
Teacher generation runs at temperature 0.3-0.5 and returns different text on
every call. Judge models are deprecated without notice — the pre-registered
``gemini-2.5-flash`` became unavailable to new API keys partway through this
project. Re-running the pipeline tomorrow produces a different corpus of
interpretations and, eventually, a different judge.

What *is* achievable is **verifiability**, and the two are worth separating:

*reproducible*
    re-running the code yields the same numbers. True of every stage that does
    not call an API: filtering, the fold assignment, the grounding checker,
    every figure.

*verifiable*
    a reader can confirm the data analysed here is the data shipped here, and
    can re-derive every deterministic step from it exactly. True of all of it,
    but only if the artifacts travel with the code and carry a checksum.

This module supplies the second. A manifest pins the content of every input by
SHA-256, alongside the git commit, the package versions and the config values
that change results — so "I ran this and got X" becomes a claim someone can
test rather than one they have to accept.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import config

log = logging.getLogger(__name__)

#: Config values that change results. Recorded by value in the manifest, so a
#: reader can see the settings a number was produced under without reading the
#: commit — and so a changed threshold is visible as a manifest diff rather
#: than being lost in a file nobody re-reads.
TRACKED_SETTINGS = (
    "SEED", "MODEL", "TEACHER_MODEL", "MIN_LINES", "MAX_SEQ_LEN",
    "MAX_POEM_TOKENS", "MIN_WORDS", "MAX_WORDS", "N_FOLDS", "FOLD_GROUP_KEY",
    "EVAL_PER_FOLD", "N_FEWSHOT", "GENERATE_MAX_ATTEMPTS",
    "GENERATE_MAX_TOTAL_ATTEMPTS", "NEAR_DUPLICATE_THRESHOLD",
    "NUMBERED_LINE_THRESHOLD", "JUDGE_TEMPERATURE", "JUDGE_SCORE_MIN",
    "JUDGE_SCORE_MAX", "MIN_JUDGE_SEPARATION",
)

#: Artifacts a result depends on. Missing files are recorded as missing rather
#: than skipped: a manifest that quietly omits an absent input would certify a
#: run that never had it.
TRACKED_ARTIFACTS = (
    "RAW_POEMS_PATH", "INTERPRETATIONS_PATH", "FOLD_ASSIGNMENT_PATH",
)


def file_digest(path: Path, chunk_size: int = 1 << 20) -> dict:
    """SHA-256 and size of one file, or a `missing` marker.

    Streamed rather than read whole: the raw corpus is 13 MB today and there is
    no reason for the check to scale with it.
    """
    path = Path(path)
    if not path.exists():
        return {"present": False}

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return {
        "present": True,
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "lines": sum(1 for _ in path.open("rb")),
    }


def git_state() -> dict:
    """The commit a run was made from, and whether the tree was dirty.

    ``dirty`` matters more than the hash. A commit id identifies code that was
    committed; it says nothing about uncommitted edits sitting in the working
    tree, and a result produced from a dirty tree cannot be traced to any
    version of the code at all.
    """
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  cwd=config.PROJECT_ROOT,
                                  timeout=10).stdout.strip() or None
        except Exception:  # pragma: no cover - git absent or not a repo
            return None

    status = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def environment() -> dict:
    """Python and the packages whose versions can change a number.

    Only the ones that can. ``scikit-learn`` decides the fold assignment and
    the attribution accuracy; ``transformers`` decides token counts and so the
    length filter. A full ``pip freeze`` would bury those among a hundred
    packages that cannot affect anything.
    """
    versions = {}
    for name in ("transformers", "scikit-learn", "numpy", "pandas", "openai"):
        try:
            from importlib.metadata import version
            versions[name] = version(name)
        except Exception:  # pragma: no cover - package absent
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def build_manifest(extra: dict | None = None) -> dict:
    """Everything needed to check a result was produced from what it claims."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "environment": environment(),
        "settings": {name: getattr(config, name, None)
                     for name in TRACKED_SETTINGS},
        "artifacts": {name: file_digest(getattr(config, name))
                      for name in TRACKED_ARTIFACTS},
        **(extra or {}),
    }


def save_manifest(path: Path | None = None, extra: dict | None = None) -> dict:
    """Write the manifest beside the results it describes."""
    path = Path(path or config.MANIFEST_PATH)
    manifest = build_manifest(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n",
                    encoding="utf-8")
    log.info("wrote %s", path)
    return manifest


def verify(path: Path | None = None) -> list[str]:
    """Re-hash the artifacts and report every mismatch.

    Returns a list of human-readable problems, empty when the data on disk is
    byte-identical to what the manifest recorded. Returning a list rather than
    raising is deliberate: a reader checking someone else's repository wants
    the full picture, not the first failure.
    """
    path = Path(path or config.MANIFEST_PATH)
    if not path.exists():
        return [f"no manifest at {path}"]

    manifest = json.loads(path.read_text(encoding="utf-8"))
    problems = []

    for name, recorded in manifest.get("artifacts", {}).items():
        current = file_digest(getattr(config, name))
        if not recorded.get("present"):
            if current.get("present"):
                problems.append(f"{name}: absent when recorded, present now")
            continue
        if not current.get("present"):
            problems.append(f"{name}: recorded but MISSING from disk")
        elif current["sha256"] != recorded["sha256"]:
            problems.append(
                f"{name}: content changed since the manifest was written "
                f"({recorded['lines']} -> {current['lines']} lines)")

    for name, recorded in manifest.get("settings", {}).items():
        current = getattr(config, name, None)
        if current != recorded:
            problems.append(f"config.{name}: manifest {recorded!r}, "
                            f"now {current!r}")

    return problems


def describe(path: Path | None = None) -> str:
    """A short human-readable provenance report for a notebook."""
    path = Path(path or config.MANIFEST_PATH)
    if not path.exists():
        return f"no manifest at {path} — run save_manifest() first"

    manifest = json.loads(path.read_text(encoding="utf-8"))
    git = manifest["git"]
    lines = [
        f"created   {manifest['created_utc']}",
        f"commit    {(git.get('commit') or '?')[:12]}"
        + ("  *** DIRTY WORKING TREE ***" if git.get("dirty") else ""),
        f"python    {manifest['environment']['python']}",
        "",
        "artifacts",
    ]
    for name, record in manifest["artifacts"].items():
        if record.get("present"):
            lines.append(f"  {name:<24} {record['sha256'][:16]}  "
                         f"{record['lines']:>6} lines")
        else:
            lines.append(f"  {name:<24} MISSING")

    problems = verify(path)
    lines += ["", "verification"]
    lines += [f"  {p}" for p in problems] if problems else \
             ["  every artifact matches its recorded digest"]
    return "\n".join(lines)
