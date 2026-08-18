"""Known-answer tests for the hypothesis machinery.

The statistics come from scipy, statsmodels and sklearn — these tests check the
project's use of them, not their internals. What can go wrong here is choosing
the wrong test, losing the pairing, pooling two judges, or reporting a null as
proof of equality; none of those raise.

Two directions are covered for every test: identical inputs must give a large
p-value, and clearly separated inputs a small one. A test that only ever passes
one way would not catch an inverted comparison.
"""

from __future__ import annotations

import config
from src.eval import metrics


# --- proportions --------------------------------------------------------------

def test_identical_proportions_give_a_large_p_value():
    result = metrics.two_proportion_test(75, 150, 75, 150)
    assert result.p_value > 0.9
    assert abs(result.effect) < 1e-9
    assert not result.significant


def test_separated_proportions_give_a_small_p_value():
    result = metrics.two_proportion_test(140, 150, 60, 150)
    assert result.p_value < 0.001
    assert result.effect > 0.5
    assert result.significant


def test_the_effect_is_the_raw_rate_difference():
    """Not an odds ratio: a reader can act on 'compliance rose 22 points' and
    cannot act on 'the odds ratio was 3.1'."""
    result = metrics.two_proportion_test(120, 150, 90, 150)
    assert abs(result.effect - (120 / 150 - 90 / 150)) < 1e-12
    assert result.effect_name == "difference in rate"


def test_the_interval_brackets_the_effect():
    result = metrics.two_proportion_test(120, 150, 90, 150)
    assert result.ci[0] < result.effect < result.ci[1]


def test_an_empty_arm_raises():
    try:
        metrics.two_proportion_test(0, 0, 10, 20)
    except AssertionError:
        return
    raise AssertionError("a proportion test ran on an empty arm")


def test_wilson_intervals_stay_inside_zero_and_one():
    """Where a trivial arm like `template` lands, and where the normal
    approximation puts a bound outside [0, 1]."""
    for successes, n in ((0, 150), (150, 150), (1, 150)):
        low, high = metrics.binomial_ci(successes, n)
        assert 0.0 <= low <= high <= 1.0, (successes, n, low, high)


# --- paired comparisons -------------------------------------------------------

def test_identical_distributions_are_not_separated():
    values = [float(i % 7) for i in range(120)]
    result = metrics.mann_whitney(values, list(values))
    assert result.p_value > 0.9
    assert not result.significant


def test_separated_distributions_are_detected():
    result = metrics.mann_whitney([8.0] * 60 + [9.0] * 60,
                                  [1.0] * 60 + [2.0] * 60)
    assert result.p_value < 0.001
    assert result.effect > 0.9          # rank-biserial near +1


def test_the_rank_biserial_sign_follows_the_argument_order():
    """An inverted comparison would report the effect backwards while the
    p-value looked identical."""
    high, low = [9.0] * 50, [1.0] * 50
    assert metrics.mann_whitney(high, low).effect > 0
    assert metrics.mann_whitney(low, high).effect < 0


def test_the_bootstrap_interval_brackets_the_mean_difference():
    differences = [0.5] * 100 + [1.5] * 100      # mean 1.0
    low, high = metrics.paired_bootstrap_ci(differences)
    assert low < 1.0 < high


def test_a_bootstrap_on_one_value_is_not_computable():
    low, high = metrics.paired_bootstrap_ci([1.0])
    assert low != low and high != high           # NaN, not a fabricated interval


# --- correlation --------------------------------------------------------------

def test_perfect_agreement_in_ranking():
    result = metrics.rank_correlation([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    assert result.effect > 0.99


def test_reversed_ranking():
    result = metrics.rank_correlation([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
    assert result.effect < -0.99


def test_too_few_points_is_not_computable():
    """H4 runs over arms, not poems — with two arms there is nothing to
    correlate, and a fabricated rho would be worse than a blank."""
    result = metrics.rank_correlation([1, 2], [2, 1])
    assert result.p_value != result.p_value


# --- agreement ----------------------------------------------------------------

def test_perfect_agreement_gives_kappa_one():
    labels = [1, 2, 3, 4, 5] * 20
    assert abs(metrics.cohens_kappa(labels, list(labels)).effect - 1.0) < 1e-9


def test_chance_agreement_gives_kappa_near_zero():
    a = [i % 2 for i in range(200)]
    b = [(i // 2) % 2 for i in range(200)]
    assert abs(metrics.cohens_kappa(a, b).effect) < 0.2


def test_kappa_is_undefined_when_neither_judge_varies():
    """Raw agreement is 100% and there is no chance level to correct against.
    1.0 would claim perfect agreement on no evidence."""
    result = metrics.cohens_kappa([9] * 50, [9] * 50)
    assert result.effect != result.effect
    assert result.detail["raw_agreement"] == 1.0


# --- position bias ------------------------------------------------------------

def test_a_consistent_judge_has_no_flips():
    result = metrics.flip_rate(["a"] * 100, ["a"] * 100)
    assert result.effect == 0.0
    assert result.detail["consistency"] == 1.0


def test_a_judge_that_always_picks_the_first_flips_every_time():
    result = metrics.flip_rate(["a"] * 100, ["b"] * 100)
    assert result.effect == 1.0


# --- win rates ----------------------------------------------------------------

def test_a_coin_flip_win_rate_is_not_significant():
    result = metrics.win_rate(75, 150)
    assert result.p_value > 0.9
    assert abs(result.effect) < 1e-9


def test_a_dominant_arm_is_detected():
    result = metrics.win_rate(140, 150)
    assert result.p_value < 0.001
    assert result.effect > 0.4


# --- reporting rules ----------------------------------------------------------

def test_a_null_never_claims_equality():
    """H2 and H3 predict nulls, which makes this load-bearing: failing to detect
    an effect at n=152 is not evidence that none exists."""
    verdict = metrics.two_proportion_test(75, 150, 75, 150).verdict()
    assert "no detected difference" in verdict
    for forbidden in ("equal", "equivalent", "no difference", "same"):
        assert forbidden not in verdict.lower()


def test_mixed_judges_cannot_be_aggregated():
    """Pooling turns a robustness check into a composite metric nobody can
    interpret, and hides a disagreement that is itself a finding."""
    results = [metrics.Result(name="H1", judge="gpt4o_mini"),
               metrics.Result(name="H1", judge="gemini_flash")]
    try:
        metrics.assert_single_judge(results)
    except AssertionError as error:
        assert "beside it" in str(error)
        return
    raise AssertionError("results from two judges were aggregated")


def test_one_judge_aggregates_fine():
    metrics.assert_single_judge([metrics.Result(name="H1", judge="gpt4o_mini"),
                                 metrics.Result(name="H2", judge="gpt4o_mini")])


def test_correction_is_attached_not_substituted():
    """Holm values are a robustness check. Two hypotheses predict nulls, where
    correction makes the predicted outcome easier to obtain."""
    results = metrics.adjust([
        metrics.two_proportion_test(140, 150, 60, 150, name="H1"),
        metrics.two_proportion_test(75, 150, 74, 150, name="H2"),
    ])
    for result in results:
        assert "p_adjusted" in result.detail
        assert result.detail["p_adjusted"] >= result.p_value
    assert not config.CORRECT_MULTIPLE_COMPARISONS


def test_the_frame_carries_effects_and_intervals_not_just_p_values():
    """A significant difference of 0.4 points is a different claim from a
    significant difference of 4, and a p-value cannot tell them apart."""
    frame = metrics.to_frame([metrics.two_proportion_test(140, 150, 60, 150,
                                                          name="H1")])
    for column in ("effect", "ci_low", "ci_high", "p_value", "verdict"):
        assert column in frame.columns
