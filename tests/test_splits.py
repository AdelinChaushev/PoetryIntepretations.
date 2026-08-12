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
