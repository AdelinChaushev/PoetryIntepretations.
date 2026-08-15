"""Partition the corpus into author-grouped cross-validation folds.

**Grouped by author, not cut at the poem level.** Poems by one author are not
independent samples — themes, diction and preoccupations recur — so training on
one Dickinson poem carries information about every other Dickinson poem. A
poem-level split would put correlated items on both sides of the partition: the
same violation as letting one patient's scans straddle train and test. This is
grouped cross-validation, as in sklearn's ``GroupKFold``.

The partitioning itself is ``sklearn.model_selection.GroupKFold`` rather than
anything hand-written. It implements exactly the heuristic this needs — groups
sorted by size descending, each assigned to the fold holding fewest samples —
and there is no credit in reimplementing a maintained, tested version of it.

Balancing is the hard part regardless. This corpus is severely skewed —
Dickinson, Byron and Shelley are roughly a third of it between them — and an
author cannot be split, so folds can never be exactly equal. That is why
:func:`assert_balanced` exists.

**On ordering:** the split happens after filtering and after teacher
generation, which is safe *here* but not safe in general. Splitting before
preprocessing matters when preprocessing **learns** from the data — a fitted
scaler, a vocabulary built from the corpus, target encoding. Nothing in this
pipeline does: the filters are per-item predicates, the tokeniser is pretrained
and never fitted on this corpus, and each interpretation depends only on its
own poem. No statistic crosses between items before the split, so no
information leaks backwards through it.

Every assertion in this module exists because the corresponding bug is
**silent**: a broken partition still produces folds, still trains, and still
yields plausible numbers. Nothing downstream would notice.
"""

from __future__ import annotations

import collections
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import config

log = logging.getLogger(__name__)


@dataclass
class Folds:
    """A grouped k-fold partition of the corpus."""

    k: int
    #: Poems held out in each fold, indexed by fold number.
    held_out: list[list[dict]]

    def training(self, fold: int) -> list[dict]:
        """Poems a model for ``fold`` may train on — every other fold."""
        return [poem
                for index, group in enumerate(self.held_out)
                if index != fold
                for poem in group]

    def fold_of(self, poem_id: str) -> int | None:
        """Which fold holds ``poem_id`` out, or None if it is not in any."""
        for index, group in enumerate(self.held_out):
            if any(poem["poem_id"] == poem_id for poem in group):
                return index
        return None

    def all_poems(self) -> list[dict]:
        return [poem for group in self.held_out for poem in group]

    def sizes(self) -> list[int]:
        return [len(group) for group in self.held_out]


# --- construction ----------------------------------------------------------

def reserve_exemplars(corpus: list[dict], n: int, seed: int) -> list[dict]:
    """Pick the fixed few-shot examples for the ``base_few`` arm.

    Chosen deterministically from (corpus, seed) rather than pinned by hand, so
    the choice is reproducible on any machine without config depending on data.

    Drawn from *distinct* authors, and their authors are excluded from the
    folds entirely by :func:`make_folds`. A Dickinson exemplar sitting in every
    prompt while Dickinson also occupies an evaluation fold would leak her
    themes through the prompt rather than through the weights — the same
    failure, by a different route.
    """
    rng = random.Random(seed)
    by_author: dict[str, list[dict]] = collections.defaultdict(list)
    for poem in corpus:
        by_author[poem["author"]].append(poem)

    authors = sorted(by_author)
    rng.shuffle(authors)

    return [rng.choice(sorted(by_author[author], key=lambda p: p["poem_id"]))
            for author in authors[:n]]


def make_folds(
    corpus: list[dict],
    k: int,
    group_key: str = "author",
    seed: int = 0,
    exclude: list[dict] | None = None,
) -> Folds:
    """Partition ``corpus`` into ``k`` folds, keeping each group intact.

    Delegates to ``sklearn.model_selection.GroupKFold``, which places the
    largest groups first into whichever fold currently holds fewest samples.
    Largest-first is what makes this work at all: assigning a 300-poem author
    after the folds have filled leaves no way to compensate.

    ``GroupKFold`` is deterministic — it does not shuffle — so the same corpus
    yields the same partition on every machine. That matters more than it
    sounds: this assignment is computed locally and shipped to Kaggle, and a
    partition that differed between them would break the held-out guarantee
    without raising anything. ``seed`` is accepted for interface consistency
    and only shuffles poems *within* a fold, never across them.
    """
    from sklearn.model_selection import GroupKFold

    excluded_groups = {poem[group_key] for poem in (exclude or [])}
    if excluded_groups:
        log.info("excluding %d group(s) reserved as exemplars: %s",
                 len(excluded_groups), ", ".join(sorted(excluded_groups)))

    # Sorted by id so the input order — and therefore the output — does not
    # depend on how the corpus happened to be read off disk.
    poems = sorted(
        (poem for poem in corpus if poem[group_key] not in excluded_groups),
        key=lambda poem: poem["poem_id"],
    )
    groups = [poem[group_key] for poem in poems]

    n_groups = len(set(groups))
    if n_groups < k:
        raise ValueError(
            f"cannot build {k} folds from {n_groups} distinct {group_key}(s): "
            f"a group cannot be split across folds"
        )

    splitter = GroupKFold(n_splits=k)
    held_out: list[list[dict]] = [[] for _ in range(k)]
    for index, (_, held_out_idx) in enumerate(
        splitter.split(poems, groups=groups)
    ):
        held_out[index] = [poems[i] for i in held_out_idx]

    rng = random.Random(seed)
    for group in held_out:
        rng.shuffle(group)

    result = Folds(k=k, held_out=held_out)
    log.info("%d folds, sizes %s (from %d %ss)",
             k, result.sizes(), n_groups, group_key)
    return result


def sample_eval_poems(
    folds: Folds,
    per_fold: int,
    min_poems_per_author: int = 2,
    seed: int = 0,
) -> list[dict]:
    """Sample the evaluation poems: ``per_fold`` from each fold's held-out set.

    Restricted to authors with at least ``min_poems_per_author`` poems, so the
    same-author swap condition is computable for every evaluation poem. Keeping
    that coverage complete keeps the paired bootstrap clean; the cost is an
    evaluation set biased toward prolific authors — which is also where
    author-prior leakage is strongest, so the bias makes the test harder rather
    than easier.
    """
    rng = random.Random(seed)
    selected: list[dict] = []

    for index, group in enumerate(folds.held_out):
        counts = collections.Counter(poem["author"] for poem in group)
        eligible = [poem for poem in group
                    if counts[poem["author"]] >= min_poems_per_author]

        if len(eligible) < per_fold:
            raise ValueError(
                f"fold {index} has only {len(eligible)} poems whose author has "
                f">= {min_poems_per_author} poems in that fold, but {per_fold} "
                f"are needed. Lower EVAL_PER_FOLD or rebalance the folds."
            )

        chosen = rng.sample(sorted(eligible, key=lambda p: p["poem_id"]), per_fold)
        for poem in chosen:
            selected.append({**poem, "fold_id": index})

    return selected


# --- assertions ------------------------------------------------------------

def assert_partition_complete(folds: Folds, corpus: list[dict],
                              exclude: list[dict] | None = None) -> None:
    """Every corpus poem is in exactly one fold, or deliberately excluded."""
    excluded_authors = {poem["author"] for poem in (exclude or [])}
    expected = {poem["poem_id"] for poem in corpus
                if poem["author"] not in excluded_authors}

    seen: list[str] = [poem["poem_id"] for poem in folds.all_poems()]
    duplicates = [pid for pid, n in collections.Counter(seen).items() if n > 1]

    assert not duplicates, f"poem(s) in more than one fold: {duplicates[:5]}"
    assert set(seen) == expected, (
        f"partition is not complete: {len(expected - set(seen))} poem(s) "
        f"missing, {len(set(seen) - expected)} unexpected"
    )


def assert_no_author_across_folds(folds: Folds) -> None:
    """No author appears in two folds.

    This is the assertion that makes it grouped cross-validation rather than
    poem-level cross-validation. Without it the pipeline runs fine and produces
    a better-looking, wrong result.
    """
    seen: dict[str, int] = {}
    for index, group in enumerate(folds.held_out):
        for poem in group:
            first = seen.setdefault(poem["author"], index)
            assert first == index, (
                f"author {poem['author']!r} appears in folds {first} and "
                f"{index} — the partition is poem-level, not author-grouped"
            )


def assert_no_leakage(folds: Folds, eval_set: list[dict]) -> None:
    """No evaluation poem appears in its own fold's training partition."""
    for poem in eval_set:
        fold = poem["fold_id"]
        training_ids = {p["poem_id"] for p in folds.training(fold)}
        assert poem["poem_id"] not in training_ids, (
            f"{poem['poem_id']} is evaluated on fold {fold} but is also in "
            f"that fold's training data"
        )


def assert_exemplars_disjoint(eval_set: list[dict],
                              exemplars: list[dict]) -> None:
    """Few-shot exemplars, and their authors, are absent from evaluation."""
    exemplar_ids = {poem["poem_id"] for poem in exemplars}
    exemplar_authors = {poem["author"] for poem in exemplars}

    for poem in eval_set:
        assert poem["poem_id"] not in exemplar_ids, (
            f"{poem['poem_id']} is both a prompt exemplar and an evaluation poem")
        assert poem["author"] not in exemplar_authors, (
            f"{poem['author']!r} supplies a prompt exemplar and also appears in "
            f"the evaluation set")


def assert_same_author_sibling_exists(eval_set: list[dict],
                                      corpus: list[dict]) -> None:
    """Every evaluation poem has another poem by the same author available.

    Without a sibling there is no ``mismatched_same_act`` condition for that
    poem, and the strict control silently degrades to partial coverage.
    """
    counts = collections.Counter(poem["author"] for poem in corpus)
    for poem in eval_set:
        assert counts[poem["author"]] >= 2, (
            f"{poem['author']!r} has only one poem in the corpus, so no "
            f"same-author swap condition exists for {poem['poem_id']}")


def assert_balanced(folds: Folds, tolerance: float) -> None:
    """Fold sizes are within ``tolerance`` of the mean."""
    sizes = folds.sizes()
    mean = sum(sizes) / len(sizes)
    worst = max(abs(size - mean) / mean for size in sizes)

    assert worst <= tolerance, (
        f"fold sizes {sizes} deviate by {worst:.0%}, over the {tolerance:.0%} "
        f"tolerance. One very prolific author is likely dominating a fold."
    )


# --- reporting and persistence --------------------------------------------

def summary(folds: Folds, eval_set: list[dict]) -> str:
    """Human-readable description of the partition."""
    lines = [f"{folds.k}-fold grouped partition "
             f"({len(folds.all_poems())} poems)"]

    by_fold = collections.Counter(poem["fold_id"] for poem in eval_set)
    for index, group in enumerate(folds.held_out):
        authors = len({poem["author"] for poem in group})
        lines.append(
            f"  fold {index}:  {len(group):>5} poems  "
            f"{authors:>4} authors  train on {len(folds.training(index)):>5}  "
            f"eval {by_fold[index]:>3}"
        )
    return "\n".join(lines)


def save(folds: Folds, eval_set: list[dict], exemplars: list[dict],
         pairs: list[dict] | None = None) -> None:
    """Write everything that crosses to Kaggle, in one call.

    Computed once, locally. Never recomputed on the GPU side: a different
    partition there would break the held-out guarantee without raising.

    Args:
        pairs: the funnel's output — each surviving poem joined with its
            interpretation. Written alongside the assignment because the two
            describe each other: an assignment names poem ids, and only this
            file says which poems those are. Shipping one without the other
            leaves the GPU side rebuilding the corpus from raw inputs, which
            agrees with the assignment right up until a filter threshold
            changes, and then disagrees silently.

    Both files land in ``config.DATA_DIR`` so a single Kaggle Dataset carries
    the whole handoff.
    """
    payload = {
        "seed": config.SEED,
        "k": folds.k,
        "group_key": config.FOLD_GROUP_KEY,
        "fold_of": {poem["poem_id"]: index
                    for index, group in enumerate(folds.held_out)
                    for poem in group},
        "eval_poem_ids": {poem["poem_id"]: poem["fold_id"] for poem in eval_set},
        "exemplar_poem_ids": [poem["poem_id"] for poem in exemplars],
    }
    config.FOLD_ASSIGNMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.FOLD_ASSIGNMENT_PATH.write_text(json.dumps(payload, indent=2))
    log.info("fold assignment written to %s (%d poems, %d eval, %d exemplars)",
             config.FOLD_ASSIGNMENT_PATH, len(payload["fold_of"]),
             len(payload["eval_poem_ids"]), len(payload["exemplar_poem_ids"]))

    if pairs is None:
        log.warning("training pairs NOT written — the fold assignment names "
                    "poem ids that nothing on the Kaggle side can resolve. "
                    "Pass pairs= to write them.")
        return

    with config.TRAINING_PAIRS_PATH.open("w", encoding="utf-8") as handle:
        for record in sorted(pairs, key=lambda p: p["poem_id"]):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("training pairs written to %s (%d records)",
             config.TRAINING_PAIRS_PATH, len(pairs))


def load_training_pairs() -> list[dict]:
    """Read the shipped training pairs.

    What Kaggle calls instead of re-running the funnel. Returns the records
    exactly as written, so the GPU side never depends on the filter thresholds
    or on having a tokeniser to re-derive them.
    """
    path = config.TRAINING_PAIRS_PATH
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8")
            if line.strip()]


def load_assignment() -> dict:
    """Read the shipped fold assignment as ``poem_id -> fold``.

    Keys are ints. JSON turns them into strings on the way out, and a lookup
    with the wrong key type returns None — which every caller then treats as
    "not in this fold" and includes the poem in training.
    """
    import json

    path = config.FOLD_ASSIGNMENT_PATH
    if not path.exists():
        return {}
    saved = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in saved.get("fold_of", {}).items()}


def require_artifacts() -> None:
    """Fail loudly, and usefully, when the shipped artifacts are not present.

    The loaders return ``[]`` and ``{}`` for a missing file, deliberately — the
    test suite runs without data. On Kaggle that leniency is a trap: a wrong
    ``POETRY_DATA_DIR`` gives zero pairs and an empty assignment, and the run
    proceeds until something much later fails for a reason that looks unrelated.

    So the notebook asserts the precondition explicitly, and when it fails this
    searches the mounted inputs for the two files and reports the directory to
    set — the fix is one line, but only if you know which line.
    """
    missing = [p for p in (config.TRAINING_PAIRS_PATH,
                           config.FOLD_ASSIGNMENT_PATH) if not p.exists()]
    if not missing:
        return

    wanted = {config.TRAINING_PAIRS_PATH.name, config.FOLD_ASSIGNMENT_PATH.name}
    roots = [Path("/kaggle/input")] if Path("/kaggle/input").is_dir() else []
    roots.append(config.PROJECT_ROOT)

    found: dict = {}
    for root in roots:
        for path in root.rglob("*"):
            if path.name in wanted and path.is_file():
                found.setdefault(path.parent, set()).add(path.name)

    complete = [d for d, names in found.items() if names == wanted]
    lines = [f"missing: {', '.join(str(p) for p in missing)}",
             f"POETRY_DATA_DIR is currently {config.DATA_DIR}"]
    if complete:
        lines.append(f"\nboth files are here — set POETRY_DATA_DIR to:\n"
                     f"    {complete[0]}")
    elif found:
        lines.append("\nfound only partially:")
        lines += [f"    {d}  ({', '.join(sorted(n))})" for d, n in found.items()]
    elif roots and roots[0].name == "input":
        available = sorted(p.name for p in roots[0].iterdir())
        lines.append(f"\nnothing found. Mounted inputs: "
                     f"{', '.join(available) or '(none)'}")
        lines.append("Attach the dataset holding training_pairs.jsonl and "
                     "folds.json, then set POETRY_DATA_DIR to its directory.")
    raise FileNotFoundError("\n".join(lines))


def load_evaluation_folds() -> dict:
    """Evaluation poem id -> the fold that holds it out, as ints.

    A *second, independent* record of the same fact ``fold_of`` carries, written
    by the same :func:`save` call. Keeping both is what lets
    :func:`training_partition` cross-check them: if they ever disagree, one of
    the two is stale and a run would train on a poem it is later evaluated on.

    Note this is per-fold, not global. An evaluation poem belonging to fold 3 is
    perfectly trainable by fold 0's model — it is generated by fold 3's adapter,
    never fold 0's. Only the 30 poems a given fold holds out are off-limits to
    that fold's run.
    """
    import json

    path = config.FOLD_ASSIGNMENT_PATH
    if not path.exists():
        return {}
    saved = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in saved.get("eval_poem_ids", {}).items()}


def training_partition(pairs: list[dict], fold: int,
                       fold_of: dict | None = None) -> list[dict]:
    """Pairs a model for ``fold`` may train on — everything that fold holds out.

    Exists because the obvious inline version is wrong in a way nothing raises:
    the pairs file carries no ``fold_id``, so ``p.get("fold_id") != fold`` is
    true for every poem and the held-out fold ends up in training. Every run
    would then be evaluated on its own training data and score better for it.

    The assertions below are the point of the function. If filtering removed
    nothing, the assignment did not apply to these pairs at all.

    **Unassigned poems are included, and that is deliberate but dangerous.** The
    exemplar authors are excluded from the folds entirely, so their poems carry
    no fold and correctly belong in every run's training set. The trouble is
    that ``mapping.get(id)`` returns ``None`` both for them and for a poem the
    assignment merely failed to cover — a truncated file, a key-type mismatch,
    a corpus regenerated after the folds were written. Those are indistinguish-
    able here, and the second silently trains on held-out data.

    So the real invariant is checked directly rather than inferred: **no
    evaluation poem may appear in any training partition**, whatever its fold
    lookup says. That catches every route to the failure at once.
    """
    mapping = load_assignment() if fold_of is None else fold_of
    assert mapping, (
        "no fold assignment available — every poem would land in training, "
        "including the fold this run is meant to hold out")

    training = [p for p in pairs if mapping.get(p["poem_id"]) != fold]
    assert len(training) < len(pairs), (
        f"filtering for fold {fold} removed nothing from {len(pairs)} pairs. "
        f"Either the assignment does not cover them, or the poem_id types "
        f"disagree — in both cases this run would train on its held-out fold.")

    # Cross-checked against the evaluation record, not inferred from `mapping`
    # alone. Only the poems THIS fold holds out are off-limits: an evaluation
    # poem belonging to fold 3 is generated by fold 3's adapter and is ordinary
    # training data here.
    held_out = {i for i, f in load_evaluation_folds().items() if f == fold}
    if held_out:
        leaked = sorted({p["poem_id"] for p in training} & held_out)
        assert not leaked, (
            f"{len(leaked)} of fold {fold}'s own EVALUATION poems are in its "
            f"training partition: {leaked[:10]}"
            f"{' ...' if len(leaked) > 10 else ''}. The fold lookup returned "
            f"{[mapping.get(i) for i in leaked[:10]]} for them, so either "
            f"fold_of and eval_poem_ids disagree or the poem_id types do. "
            f"Training on these would evaluate the model on its own training "
            f"data and inflate every grounding number.")

    unassigned = {p["poem_id"] for p in training} - set(mapping)
    if unassigned:
        log.info("%d poems carry no fold and are trained on by every run "
                 "(the exemplar authors, held out of the folds by design)",
                 len(unassigned))

    log.info("fold %d held out: training on %d of %d pairs",
             fold, len(training), len(pairs))
    return training
