"""Tests for the author-grouped fold partition.

Every failure this file guards against is silent. A poem-level split still
produces folds, still trains, and still yields plausible numbers — better ones,
in fact, which is precisely why nothing downstream would flag it.
"""

from __future__ import annotations

import pytest

from src.data import splits


def make_corpus(sizes: dict[str, int]) -> list[dict]:
    """Build a corpus with a given author -> poem-count shape."""
    return [
        {
            "poem_id": f"{author}-{i}",
            "title": f"{author} {i}",
            "author": author,
            "lines": ["a line", "another line"],
            "linecount": 12,
        }
        for author, count in sizes.items()
        for i in range(count)
    ]


#: Mirrors the real corpus: a few enormous collections and a long tail.
SKEWED = make_corpus({
    "Dickinson": 300, "Byron": 240, "Shelley": 200,
    "Shakespeare": 160, "Clare": 130, "Thomas": 129, "Burns": 82,
    **{f"Minor{i}": 6 for i in range(40)},
})


# --- the assertion that defines grouped CV ---------------------------------

def test_no_author_appears_in_two_folds():
    folds = splits.make_folds(SKEWED, k=5, seed=42)
    splits.assert_no_author_across_folds(folds)


def test_assert_no_author_across_folds_catches_a_poem_level_split():
    """The guard must actually fire — a passing assertion that cannot fail
    is worse than no assertion, because it reads as protection."""
    poems = make_corpus({"Dickinson": 4})
    leaky = splits.Folds(k=2, held_out=[poems[:2], poems[2:]])

    with pytest.raises(AssertionError, match="appears in folds"):
        splits.assert_no_author_across_folds(leaky)


# --- partition integrity ---------------------------------------------------

def test_partition_is_complete_and_disjoint():
    folds = splits.make_folds(SKEWED, k=5, seed=42)
    splits.assert_partition_complete(folds, SKEWED)
    assert sum(folds.sizes()) == len(SKEWED)


def test_training_set_excludes_the_held_out_fold():
    folds = splits.make_folds(SKEWED, k=5, seed=42)

    for fold in range(5):
        held = {p["poem_id"] for p in folds.held_out[fold]}
        train = {p["poem_id"] for p in folds.training(fold)}
        assert held.isdisjoint(train)
        assert len(held) + len(train) == len(SKEWED)


def test_folds_are_balanced_despite_severe_skew():
    """Dickinson alone is 22% of this corpus and cannot be split."""
    folds = splits.make_folds(SKEWED, k=5, seed=42)
    splits.assert_balanced(folds, tolerance=0.25)


def test_assert_balanced_rejects_a_lopsided_partition():
    poems = make_corpus({"A": 90, "B": 10})
    lopsided = splits.Folds(k=2, held_out=[poems[:90], poems[90:]])

    with pytest.raises(AssertionError, match="deviate"):
        splits.assert_balanced(lopsided, tolerance=0.25)


def test_too_few_authors_for_k_folds_raises():
    with pytest.raises(ValueError, match="cannot be split across folds"):
        splits.make_folds(make_corpus({"A": 50, "B": 50}), k=5)


# --- determinism -----------------------------------------------------------

def test_partition_is_identical_across_calls():
    """The assignment is computed locally and shipped to Kaggle.

    If it differed between machines the held-out guarantee would break with
    nothing raised, and every grounding number would quietly improve.
    """
    first = splits.make_folds(SKEWED, k=5, seed=42)
    second = splits.make_folds(SKEWED, k=5, seed=42)

    assert [sorted(p["poem_id"] for p in g) for g in first.held_out] == \
           [sorted(p["poem_id"] for p in g) for g in second.held_out]


def test_partition_does_not_depend_on_input_order():
    """Reading the corpus off disk in a different order must not move poems."""
    forward = splits.make_folds(SKEWED, k=5, seed=42)
    backward = splits.make_folds(list(reversed(SKEWED)), k=5, seed=42)

    assert [sorted(p["poem_id"] for p in g) for g in forward.held_out] == \
           [sorted(p["poem_id"] for p in g) for g in backward.held_out]


# --- exemplars -------------------------------------------------------------

def test_exemplars_come_from_distinct_authors():
    exemplars = splits.reserve_exemplars(SKEWED, n=3, seed=42)
    assert len({p["author"] for p in exemplars}) == 3


def test_exemplar_authors_are_excluded_from_every_fold():
    """Reserving the poems is not enough — the authors must go too.

    A Dickinson exemplar in every prompt while Dickinson also sits in an
    evaluation fold leaks her themes through the prompt instead of the weights.
    """
    exemplars = splits.reserve_exemplars(SKEWED, n=3, seed=42)
    folds = splits.make_folds(SKEWED, k=5, seed=42, exclude=exemplars)

    fold_authors = {p["author"] for p in folds.all_poems()}
    for exemplar in exemplars:
        assert exemplar["author"] not in fold_authors


def test_assert_exemplars_disjoint_catches_a_shared_author():
    exemplars = [SKEWED[0]]
    eval_set = [{**SKEWED[1], "fold_id": 0}]  # same author, different poem

    with pytest.raises(AssertionError, match="supplies a prompt exemplar"):
        splits.assert_exemplars_disjoint(eval_set, exemplars)


# --- evaluation sampling ---------------------------------------------------

def test_eval_set_has_the_expected_size_and_tags_folds():
    folds = splits.make_folds(SKEWED, k=5, seed=42)
    eval_set = splits.sample_eval_poems(folds, per_fold=10, seed=42)

    assert len(eval_set) == 50
    assert sorted({p["fold_id"] for p in eval_set}) == [0, 1, 2, 3, 4]


def test_every_eval_poem_was_held_out_by_its_own_fold():
    """The guarantee the whole design rests on."""
    folds = splits.make_folds(SKEWED, k=5, seed=42)
    eval_set = splits.sample_eval_poems(folds, per_fold=10, seed=42)

    splits.assert_no_leakage(folds, eval_set)


def test_every_eval_poem_has_a_same_author_sibling():
    folds = splits.make_folds(SKEWED, k=5, seed=42)
    eval_set = splits.sample_eval_poems(folds, per_fold=10,
                                        min_poems_per_author=2, seed=42)

    splits.assert_same_author_sibling_exists(eval_set, SKEWED)


def test_single_poem_authors_are_never_sampled_for_evaluation():
    """Singletons have no sibling, so they cannot carry the strict control.

    The corpus mixes multi-poem and single-poem authors, as the real one does:
    every fold needs *some* eligible poems, since grouping puts all of an
    author's work in one fold.
    """
    corpus = make_corpus({
        **{f"Multi{i}": 8 for i in range(9)},
        **{f"Solo{i}": 1 for i in range(30)},
    })
    folds = splits.make_folds(corpus, k=3, seed=42)
    eval_set = splits.sample_eval_poems(folds, per_fold=2,
                                        min_poems_per_author=2, seed=42)

    assert all(not p["author"].startswith("Solo") for p in eval_set)
    splits.assert_same_author_sibling_exists(eval_set, corpus)


def test_too_few_eligible_poems_raises_rather_than_silently_shrinking():
    """A short evaluation set would weaken every CI without saying so."""
    corpus = make_corpus({"A": 10, "B": 10, "C": 10})
    folds = splits.make_folds(corpus, k=3, seed=42)

    with pytest.raises(ValueError, match="are needed"):
        splits.sample_eval_poems(folds, per_fold=99, seed=42)


# --- the Kaggle handoff -------------------------------------------------------

def test_save_writes_both_artifacts(tmp_path, monkeypatch):
    """The assignment names poem ids; only the pairs file says which poems those
    are. Shipping one without the other leaves the GPU side rebuilding the
    corpus from raw inputs."""
    import json

    import config
    from src.data import splits

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "FOLD_ASSIGNMENT_PATH", tmp_path / "folds.json")
    monkeypatch.setattr(config, "TRAINING_PAIRS_PATH", tmp_path / "pairs.jsonl")

    # Two authors minimum: a group is never split across folds, so k folds
    # need k distinct authors.
    pairs = [{"poem_id": i, "author": "AB"[i % 2], "title": "t",
              "lines": ["x"], "linecount": 1, "interpretation": f"about {i}"}
             for i in range(1, 7)]
    folds = splits.make_folds(pairs, 2, group_key="author", seed=0)
    splits.save(folds, [], [], pairs=pairs)

    assert config.FOLD_ASSIGNMENT_PATH.exists()
    assert config.TRAINING_PAIRS_PATH.exists()
    written = splits.load_training_pairs()
    assert len(written) == len(pairs)
    assert [p["poem_id"] for p in written] == sorted(p["poem_id"] for p in pairs)


def test_saved_pairs_keep_their_interpretations(tmp_path, monkeypatch):
    """Without the interpretation there is nothing to train on, and the failure
    would only surface on the GPU."""
    import config
    from src.data import splits

    monkeypatch.setattr(config, "FOLD_ASSIGNMENT_PATH", tmp_path / "folds.json")
    monkeypatch.setattr(config, "TRAINING_PAIRS_PATH", tmp_path / "pairs.jsonl")
    pairs = [{"poem_id": 1, "author": "A", "title": "t", "lines": ["x"],
              "linecount": 1, "interpretation": "the interpretation"},
             {"poem_id": 2, "author": "B", "title": "t", "lines": ["y"],
              "linecount": 1, "interpretation": "another"}]
    folds = splits.make_folds(pairs, 2, group_key="author", seed=0)
    splits.save(folds, [], [], pairs=pairs)
    assert splits.load_training_pairs()[0]["interpretation"] == "the interpretation"


def test_missing_pairs_warns_rather_than_silently_skipping(tmp_path, monkeypatch,
                                                           caplog):
    import config
    from src.data import splits

    monkeypatch.setattr(config, "FOLD_ASSIGNMENT_PATH", tmp_path / "folds.json")
    monkeypatch.setattr(config, "TRAINING_PAIRS_PATH", tmp_path / "pairs.jsonl")
    pairs = [{"poem_id": 1, "author": "A", "title": "t", "lines": ["x"],
              "linecount": 1, "interpretation": "x"},
             {"poem_id": 2, "author": "B", "title": "t", "lines": ["y"],
              "linecount": 1, "interpretation": "y"}]
    folds = splits.make_folds(pairs, 2, group_key="author", seed=0)
    with caplog.at_level("WARNING"):
        splits.save(folds, [], [])
    assert "training pairs NOT written" in caplog.text


# --- the training partition ---------------------------------------------------

def test_training_partition_excludes_the_held_out_fold():
    from src.data import splits

    pairs = [{"poem_id": i, "author": "A"} for i in range(1, 7)]
    fold_of = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    training = splits.training_partition(pairs, fold=0, fold_of=fold_of)
    assert {p["poem_id"] for p in training} == {3, 4, 5, 6}


def test_partition_raises_when_the_filter_removes_nothing():
    """The bug this replaced: the pairs file carries no fold_id, so an inline
    `p.get("fold_id") != fold` was true for every poem and every run trained on
    its own held-out fold. Silently."""
    from src.data import splits

    pairs = [{"poem_id": i, "author": "A"} for i in range(1, 4)]
    try:
        splits.training_partition(pairs, fold=99, fold_of={1: 0, 2: 0, 3: 0})
    except AssertionError as error:
        assert "removed nothing" in str(error)
        return
    raise AssertionError("a no-op fold filter was accepted")


def test_partition_raises_without_an_assignment():
    from src.data import splits

    try:
        splits.training_partition([{"poem_id": 1}], fold=0, fold_of={})
    except AssertionError:
        return
    raise AssertionError("filtering proceeded with no fold assignment")


def test_assignment_keys_are_integers():
    """JSON stringifies dict keys. A lookup with the wrong type returns None,
    which every caller reads as 'not in this fold'."""
    from src.data import splits

    mapping = splits.load_assignment()
    if mapping:
        assert all(isinstance(k, int) for k in mapping)


# --- a fold never trains on the poems it holds out ----------------------------

def test_partition_rejects_a_poem_this_fold_holds_out(monkeypatch):
    """The invariant the whole held-out guarantee rests on, asserted directly.

    Checked against the evaluation record rather than inferred from the fold
    lookup, because ``mapping.get(id)`` returns None both for a deliberately
    unassigned exemplar author and for a poem the assignment failed to cover.
    Only the second is a bug, and the two are indistinguishable at the lookup.
    """
    from src.data import splits

    pairs = [{"poem_id": i} for i in range(1, 6)]
    # Poem 4 is evaluated by fold 0, but `mapping` puts it in fold 1, so a
    # fold-0 run would happily train on the poem it is later judged on.
    monkeypatch.setattr(splits, "load_evaluation_folds", lambda: {4: 0})

    try:
        splits.training_partition(pairs, fold=0,
                                  fold_of={1: 0, 2: 1, 3: 1, 4: 1, 5: 1})
    except AssertionError as error:
        assert "EVALUATION" in str(error)
        return
    raise AssertionError("a fold trained on a poem it holds out")


def test_partition_catches_the_key_type_mismatch(monkeypatch):
    """The failure ``load_assignment``'s docstring warns about, made to fail.

    String keys make every int lookup return None, every poem reads as "not in
    this fold", and the held-out poems land in training. The older "removed
    nothing" assertion does not fire, because one poem still happens to filter.
    """
    from src.data import splits

    pairs = [{"poem_id": i} for i in range(1, 6)]
    monkeypatch.setattr(splits, "load_evaluation_folds", lambda: {2: 0, 3: 0})

    try:
        splits.training_partition(pairs, fold=0,
                                  fold_of={"2": 0, "3": 0, "4": 1, 1: 0})
    except AssertionError as error:
        assert "EVALUATION" in str(error)
        return
    raise AssertionError("a stringified assignment was accepted")


def test_other_folds_evaluation_poems_are_trainable(monkeypatch):
    """Must NOT raise. An evaluation poem belonging to fold 3 is generated by
    fold 3's adapter, so fold 0 training on it breaks nothing — over-strictness
    here would throw away 120 of the 150 for no reason."""
    from src.data import splits

    pairs = [{"poem_id": i} for i in range(1, 6)]
    monkeypatch.setattr(splits, "load_evaluation_folds", lambda: {2: 3, 5: 0})

    training = splits.training_partition(pairs, fold=0,
                                         fold_of={1: 1, 2: 3, 4: 1, 5: 0})
    assert 2 in {p["poem_id"] for p in training}     # fold 3's, trainable here
    assert 5 not in {p["poem_id"] for p in training}  # fold 0's own, excluded


def test_unassigned_exemplar_authors_are_still_allowed(monkeypatch):
    """Also must not raise: exemplar-author poems carry no fold at all and
    belong in every run's training set, because they are never evaluated on."""
    from src.data import splits

    pairs = [{"poem_id": i} for i in range(1, 6)]
    monkeypatch.setattr(splits, "load_evaluation_folds", lambda: {5: 0})

    training = splits.training_partition(pairs, fold=0,
                                         fold_of={1: 1, 2: 0, 4: 1, 5: 0})
    assert 3 in {p["poem_id"] for p in training}     # no fold — an exemplar author


def test_evaluation_fold_keys_are_integers():
    from src.data import splits

    folds = splits.load_evaluation_folds()
    if folds:
        assert all(isinstance(i, int) for i in folds)


def test_the_two_fold_records_agree():
    """``fold_of`` and ``eval_poem_ids`` are written by one save() call and must
    never drift; a disagreement means one is stale."""
    from src.data import splits

    mapping, evaluation = splits.load_assignment(), splits.load_evaluation_folds()
    if mapping and evaluation:
        assert all(mapping.get(i) == f for i, f in evaluation.items())


def test_no_fold_trains_on_its_own_evaluation_poems():
    """End-to-end on the shipped assignment, for all five folds."""
    import config
    from src.data import splits

    pairs = splits.load_training_pairs()
    evaluation = splits.load_evaluation_folds()
    if not pairs or not evaluation:
        return

    for fold in range(config.N_FOLDS):
        trained = {p["poem_id"] for p in splits.training_partition(pairs, fold)}
        assert not trained & {i for i, f in evaluation.items() if f == fold}
