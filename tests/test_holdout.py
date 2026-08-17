"""Tests for the outer holdout split — the design that replaces 5-fold.

Cross-validation was applied to the TEST partition, which forced five final
models where one would do and produced no single loadable artefact. This module
covers the replacement: one author-disjoint test set fixed before any
hyperparameter is chosen, k-fold confined to tuning where it belongs.

Every assertion here guards a silent failure. A broken outer split still trains,
still generates, and still yields plausible numbers.
"""

from __future__ import annotations

import collections

import config
from src.data import splits


def corpus(sizes: dict[str, int]) -> list[dict]:
    """A synthetic corpus: author -> how many poems they have."""
    poems, pid = [], 0
    for author, n in sizes.items():
        for _ in range(n):
            pid += 1
            poems.append({"poem_id": pid, "author": author, "title": f"T{pid}",
                          "linecount": 12, "lines": ["a line"] * 12,
                          "interpretation": "an interpretation"})
    return poems


SKEWED = corpus({"Prolific A": 100, "Prolific B": 80, "Mid": 20,
                 **{f"Small {i}": 3 for i in range(12)},
                 **{f"Solo {i}": 1 for i in range(5)}})


# --- the split itself ---------------------------------------------------------

def test_test_and_pool_share_no_author():
    """The load-bearing property. A poem-level split leaves the model trained on
    271 poems by an author it is then tested on."""
    test, pool = splits.holdout_split(SKEWED, test_size=20)
    assert not ({p["author"] for p in test} & {p["author"] for p in pool})


def test_test_and_pool_partition_the_corpus():
    test, pool = splits.holdout_split(SKEWED, test_size=20)
    assert len(test) + len(pool) == len(SKEWED)
    assert not ({p["poem_id"] for p in test} & {p["poem_id"] for p in pool})


def test_small_authors_are_taken_first():
    """Taking prolific authors would reach the same test size while gutting the
    training pool: 16 authors own 66% of the real corpus."""
    test, pool = splits.holdout_split(SKEWED, test_size=20)
    assert "Prolific A" not in {p["author"] for p in test}
    assert "Prolific B" not in {p["author"] for p in test}
    assert len(pool) / len(SKEWED) > 0.8


def test_authors_with_one_poem_are_never_in_test():
    """The swap test needs a same-author sibling for every test poem."""
    test, _ = splits.holdout_split(SKEWED, test_size=20)
    counts = collections.Counter(p["author"] for p in test)
    assert all(n >= 2 for n in counts.values())
    assert not any(a.startswith("Solo") for a in counts)


def test_the_split_is_deterministic():
    """It is quoted in the report and probed for contamination; it cannot drift
    between runs or between machines."""
    a, _ = splits.holdout_split(SKEWED, test_size=20, seed=config.SEED)
    b, _ = splits.holdout_split(SKEWED, test_size=20, seed=config.SEED)
    assert [p["poem_id"] for p in a] == [p["poem_id"] for p in b]


def test_exemplar_authors_are_excluded_from_both():
    exemplars = [SKEWED[0]]
    test, pool = splits.holdout_split(SKEWED, test_size=20, exclude=exemplars)
    author = exemplars[0]["author"]
    assert author not in {p["author"] for p in test}


def test_an_impossible_test_size_raises():
    """Better than silently returning a smaller test set than asked for."""
    tiny = corpus({"A": 2, "B": 2})
    try:
        splits.holdout_split(tiny, test_size=500)
    except AssertionError as error:
        assert "needed for the test set" in str(error)
        return
    raise AssertionError("an unreachable test size was accepted")


# --- the assertions -----------------------------------------------------------

def test_a_shared_author_is_caught():
    test, pool = splits.holdout_split(SKEWED, test_size=20)
    try:
        splits.assert_holdout_disjoint(test, pool + [test[0]])
    except AssertionError as error:
        assert "author priors" in str(error) or "BOTH" in str(error)
        return
    raise AssertionError("a shared author passed the disjointness check")


def test_an_exemplar_in_the_test_set_is_caught():
    """base_few would be shown the answer inside its own prompt."""
    test, pool = splits.holdout_split(SKEWED, test_size=20)
    try:
        splits.assert_holdout_disjoint(test, pool, exemplars=[test[0]])
    except AssertionError as error:
        assert "exemplar" in str(error)
        return
    raise AssertionError("an exemplar that is also a test poem passed")


def test_a_test_poem_without_a_sibling_is_caught():
    lonely = [{"poem_id": 999, "author": "Nobody", "title": "T",
               "linecount": 8, "lines": [], "interpretation": ""}]
    try:
        splits.assert_test_has_siblings(lonely)
    except AssertionError as error:
        assert "same-author sibling" in str(error)
        return
    raise AssertionError("a test poem with no sibling passed")


# --- k-fold stays in tuning ---------------------------------------------------

def test_tuning_folds_never_contain_a_test_poem():
    """Hyperparameters chosen with the test set in view make every number
    reported from it optimistic — the +11% selection bias, applied to the
    headline result instead of to perplexity."""
    test, pool = splits.holdout_split(SKEWED, test_size=20)
    folds = splits.tuning_folds(pool, k=3, seed=config.SEED)
    splits.assert_tuning_never_sees_test(folds, test)


def test_a_test_poem_leaking_into_tuning_is_caught():
    test, pool = splits.holdout_split(SKEWED, test_size=20)
    folds = splits.tuning_folds(pool, k=3, seed=config.SEED)
    folds.held_out[0].append(test[0])
    try:
        splits.assert_tuning_never_sees_test(folds, test)
    except AssertionError as error:
        assert "reported from" in str(error)
        return
    raise AssertionError("a test poem inside a tuning fold passed")


def test_the_tuning_subsample_takes_whole_authors():
    """A subsample that split an author would break the grouping the folds
    built over it depend on."""
    _, pool = splits.holdout_split(SKEWED, test_size=20)
    folds = splits.tuning_folds(pool, k=3, subsample=60, seed=config.SEED)

    used = collections.Counter(p["author"] for p in folds.all_poems())
    whole = collections.Counter(p["author"] for p in pool)
    for author, n in used.items():
        assert n == whole[author], f"{author} was split by the subsample"


def test_tuning_folds_keep_authors_within_one_fold():
    _, pool = splits.holdout_split(SKEWED, test_size=20)
    folds = splits.tuning_folds(pool, k=3, seed=config.SEED)
    splits.assert_no_author_across_folds(folds)


# --- validation must not reuse tuning data ------------------------------------

def test_validation_prefers_authors_tuning_never_saw():
    """The final model's stopping point should not be chosen on data whose
    hyperparameters were also chosen on it. The optimism is mild and lands on
    the validation number rather than the test number — but 1,380 pool poems
    sit outside the tuning subsample, so avoiding it is free."""
    from src.data import filter as F
    from src.train import dataset, loop

    tokenizer = F.get_tokenizer()
    test, pool = splits.holdout_split(SKEWED, test_size=20)
    folds = splits.tuning_folds(pool, k=3, subsample=40, seed=config.SEED)
    tuned_authors = {p["author"] for p in folds.all_poems()}
    untouched = {p["author"] for p in pool} - tuned_authors
    assert untouched, "the fixture needs authors outside the tuning subsample"

    examples = dataset.build_dataset(pool, tokenizer)
    _, validation = loop.split_validation(examples, pool,
                                          prefer_unused=untouched)

    author_of = {p["poem_id"]: p["author"] for p in pool}
    chosen = {author_of[e["poem_id"]] for e in validation}
    assert chosen <= untouched, (
        f"validation drew on authors tuning already used: {chosen - untouched}")


def test_validation_still_works_without_the_preference():
    """prefer_unused is optional — the fold-based path passes nothing."""
    from src.data import filter as F
    from src.train import dataset, loop

    tokenizer = F.get_tokenizer()
    _, pool = splits.holdout_split(SKEWED, test_size=20)
    examples = dataset.build_dataset(pool, tokenizer)
    train, validation = loop.split_validation(examples, pool)
    assert train and validation


# --- persistence --------------------------------------------------------------

def test_the_holdout_round_trips(tmp_path, monkeypatch):
    """Keys are ints. JSON stringifies them, and a lookup with the wrong type
    returns nothing — which every caller reads as 'not in the test set' and
    quietly trains on it."""
    monkeypatch.setattr(config, "HOLDOUT_PATH", tmp_path / "holdout.json")
    test, pool = splits.holdout_split(SKEWED, test_size=20)
    folds = splits.tuning_folds(pool, k=3, seed=config.SEED)

    splits.save_holdout(test, pool, [], folds)
    loaded = splits.load_holdout()

    assert loaded["test"] == {p["poem_id"] for p in test}
    assert loaded["pool"] == {p["poem_id"] for p in pool}
    assert all(isinstance(i, int) for i in loaded["test"])
    assert loaded["tuning_k"] == 3


def test_saving_runs_every_assertion(tmp_path, monkeypatch):
    """save_holdout is the last gate before a file exists that everything
    downstream trusts."""
    monkeypatch.setattr(config, "HOLDOUT_PATH", tmp_path / "holdout.json")
    test, pool = splits.holdout_split(SKEWED, test_size=20)
    folds = splits.tuning_folds(pool, k=3, seed=config.SEED)
    folds.held_out[0].append(test[0])          # leak a test poem into tuning

    try:
        splits.save_holdout(test, pool, [], folds)
    except AssertionError:
        assert not (tmp_path / "holdout.json").exists(), "wrote despite failing"
        return
    raise AssertionError("a leaking split was written to disk")


def test_pool_partition_refuses_without_a_holdout(tmp_path, monkeypatch):
    """No holdout means every poem looks trainable, including the test set."""
    monkeypatch.setattr(config, "HOLDOUT_PATH", tmp_path / "absent.json")
    try:
        splits.pool_partition(SKEWED)
    except AssertionError as error:
        assert "refusing" in str(error)
        return
    raise AssertionError("the whole corpus was treated as trainable")


def test_pool_partition_catches_a_no_op_filter(tmp_path, monkeypatch):
    """If the ids do not match, filtering removes nothing and the run trains on
    its own test set — the same silent failure training_partition guards."""
    monkeypatch.setattr(config, "HOLDOUT_PATH", tmp_path / "holdout.json")
    (tmp_path / "holdout.json").write_text(
        '{"test_poem_ids": [99999], "pool_poem_ids": [1]}')
    try:
        splits.pool_partition(SKEWED)
    except AssertionError as error:
        assert "own test set" in str(error)
        return
    raise AssertionError("a no-op filter passed")


def test_untouched_authors_excludes_everything_tuning_saw(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HOLDOUT_PATH", tmp_path / "holdout.json")
    test, pool = splits.holdout_split(SKEWED, test_size=20)
    folds = splits.tuning_folds(pool, k=3, subsample=40, seed=config.SEED)
    splits.save_holdout(test, pool, [], folds)

    untouched = splits.untouched_authors(SKEWED)
    tuned = {p["author"] for p in folds.all_poems()}
    assert untouched and not (untouched & tuned)


# --- CV selection --------------------------------------------------------------

def cv_row(rank, lr, fold, loss):
    return {"run": f"cv_r{rank}_lr{lr:g}_f{fold}", "model": config.MODEL,
            "rank": rank, "learning_rate": lr, "trainable_params": 700_000,
            config.SWEEP_SELECTION_METRIC: loss}


def test_cv_selects_on_the_mean_not_the_best_fold():
    """Taking the single best fold selects the luckiest slice of authors —
    exactly the fragility CV exists to remove. With three folds per config,
    best-of-three is a noticeably optimistic estimate."""
    from src.train import sweep

    rows = []
    # A: consistently good (mean 1.60). B: one lucky fold, bad mean (1.75).
    for fold, loss in enumerate([1.60, 1.60, 1.60]):
        rows.append(cv_row(8, 1e-4, fold, loss))
    for fold, loss in enumerate([1.40, 1.90, 1.95]):
        rows.append(cv_row(16, 1e-4, fold, loss))

    winner = sweep.select_winner_cv(rows)
    assert winner["rank"] == 8, "selected the lucky fold rather than the mean"


def test_a_configuration_missing_folds_is_excluded():
    """A mean over two folds is not comparable with a mean over three."""
    from src.train import sweep

    rows = [cv_row(8, 1e-4, f, 1.70) for f in range(config.TUNING_FOLDS)]
    rows.append(cv_row(4, 1e-4, 0, 0.90))          # one fold only, very low
    winner = sweep.select_winner_cv(rows)
    assert winner["rank"] == 8


def test_cv_refuses_when_nothing_is_complete():
    from src.train import sweep

    try:
        sweep.select_winner_cv([cv_row(8, 1e-4, 0, 1.7)])
    except AssertionError as error:
        assert "folds recorded" in str(error)
        return
    raise AssertionError("selection proceeded with no complete configuration")


def test_cv_ties_break_toward_the_smaller_rank():
    """Hu et al.'s saturation direction, so the tiebreak cannot quietly favour
    the outcome."""
    from src.train import sweep

    rows = ([cv_row(16, 1e-4, f, 1.60) for f in range(config.TUNING_FOLDS)]
            + [cv_row(4, 1e-4, f, 1.60) for f in range(config.TUNING_FOLDS)])
    assert sweep.select_winner_cv(rows)["rank"] == 4


def test_cv_specs_cover_every_config_on_every_fold():
    from src.train import sweep

    specs = sweep.cv_specs()
    assert len(specs) == len(sweep.tuning_grid()) * config.TUNING_FOLDS
    assert len({s["run"] for s in specs}) == len(specs)


def test_the_final_run_has_no_fold_suffix():
    """There is exactly one, which is what the README's load snippet promises
    and what the fold design could not provide."""
    from src.train import sweep

    spec = sweep.final_spec({"rank": 8, "learning_rate": 1e-4})
    assert spec["run"] == "lora_r8"
    assert "fold" not in spec["run"]
    assert config.run_adapter_dir(spec["run"]).name.endswith("lora_r8")


def test_the_final_run_refuses_a_pool_overlapping_test():
    """The check that catches a split rebuilt inconsistently — the run would
    otherwise train on the data it is about to be measured by."""
    import inspect

    from src.train import sweep

    source = inspect.getsource(sweep.run_final)
    assert "pool and the test set overlap" in source
    assert "splits.test_partition" in source
