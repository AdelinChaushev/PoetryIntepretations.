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


# --- model-free summary -------------------------------------------------------

def corpus_fixture():
    return [{"poem_id": 1, "author": "A", "title": "T",
             "lines": ["the cat sat down", "upon the mat today"]},
            {"poem_id": 2, "author": "B", "title": "U",
             "lines": ["a bird flew high", "over the quiet hill"]}]


def outputs_fixture():
    grounded = ('1. Central idea - about cats.\n'
                '2. Key images - "the cat sat down" and "upon the mat today".\n'
                '3. Tone - calm.\n'
                '4. Interpretive claim - it is about rest.')
    hallucinated = ('1. Central idea - about dogs.\n'
                    '2. Key images - "the dog ran away fast".\n'
                    '3. Tone - brisk.\n'
                    '4. Interpretive claim - it is about motion.')
    silent = "It is a poem about many things, broadly speaking and at length."
    return [{"poem_id": 1, "arm": "good", "interpretation": grounded},
            {"poem_id": 2, "arm": "good", "interpretation": grounded.replace(
                "the cat sat down", "a bird flew high").replace(
                "upon the mat today", "over the quiet hill")},
            {"poem_id": 1, "arm": "bad", "interpretation": hallucinated},
            {"poem_id": 2, "arm": "bad", "interpretation": silent}]


def test_grounding_is_all_or_nothing_per_interpretation():
    """The pre-registered H2 quantity. An arm that quotes more often is held to
    a stricter bar, which is why the per-quote rate is reported beside it."""
    from src.eval import metrics

    s = metrics.model_free_summary(outputs_fixture(), corpus_fixture(),
                                   arms=("good", "bad"))
    good = s.set_index("arm").loc["good"]
    assert good["grounded"] == 2 and good["grounding_rate"] == 1.0
    assert s.set_index("arm").loc["bad"]["grounded"] == 0


def test_quoting_nothing_counts_as_ungrounded():
    """It dodged the question rather than answering it. Treating it as a pass
    would reward exactly the vague, unquotable output the project detects."""
    from src.eval import metrics

    s = metrics.model_free_summary(outputs_fixture(), corpus_fixture(),
                                   arms=("bad",)).iloc[0]
    assert s["quoted_nothing"] == 1
    assert s["grounding_rate"] == 0.0


def test_an_output_with_no_poem_raises():
    """Its quotations cannot be checked, and scoring it against nothing would
    silently record a hallucination."""
    from src.eval import metrics

    stray = outputs_fixture() + [{"poem_id": 99, "arm": "good",
                                  "interpretation": "x"}]
    try:
        metrics.model_free_summary(stray, corpus_fixture(), arms=("good",))
    except AssertionError as error:
        assert "no poem in the corpus" in str(error)
        return
    raise AssertionError("an output with no poem was summarised anyway")


def test_comparison_runs_on_counts_not_rounded_rates():
    """A z-test fed a rounded proportion is a z-test on the wrong number."""
    from src.eval import metrics

    s = metrics.model_free_summary(outputs_fixture(), corpus_fixture(),
                                   arms=("good", "bad"))
    results = metrics.compare_to_baseline(s, "grounding_rate", baseline="bad")
    assert len(results) == 1
    assert results[0].n == 4                      # 2 + 2, the integer counts
    assert results[0].effect == 1.0               # 100% vs 0%


def test_comparing_an_unknown_column_raises():
    from src.eval import metrics

    s = metrics.model_free_summary(outputs_fixture(), corpus_fixture(),
                                   arms=("good", "bad"))
    try:
        metrics.compare_to_baseline(s, "median_words", baseline="bad")
    except AssertionError as error:
        assert "count column" in str(error)
        return
    raise AssertionError("a column with no success count was tested anyway")


# --- H3 and H4 ----------------------------------------------------------------

def scored_fixture(arm, matched, random_, same_author, judge="j1"):
    """One arm's swap-test records: three conditions per poem."""
    rows = []
    for i, (m, r, s) in enumerate(zip(matched, random_, same_author)):
        for condition, score in (("matched", m), ("mismatched_random", r),
                                 ("mismatched_same_author", s)):
            rows.append({"judge": judge, "arm": arm, "poem_id": i,
                         "condition": condition, "score": score})
    return rows


def test_h3_detects_a_widened_gap():
    from src.eval import metrics

    weak = scored_fixture("base_few", [5] * 20, [4] * 20, [4] * 20)
    strong = scored_fixture("lora_r8", [9] * 20, [2] * 20, [2] * 20)
    r = metrics.h3_grounding_gap(weak + strong, arms=["lora_r8"])[0]
    assert r.detail["mean_arm"] == 7 and r.detail["mean_baseline"] == 1
    assert r.p_value < 0.001
    assert r.effect > 0.9


def test_h3_reports_no_difference_when_gaps_match():
    """A null must read as 'no detected difference', never as equality."""
    from src.eval import metrics

    a = scored_fixture("base_few", [7] * 20, [3] * 20, [3] * 20)
    b = scored_fixture("lora_r8", [7] * 20, [3] * 20, [3] * 20)
    r = metrics.h3_grounding_gap(a + b, arms=["lora_r8"])[0]
    assert r.p_value > 0.05
    assert "no detected difference" in r.verdict()


def test_h3_can_use_the_strict_condition():
    """The poem-level gap is the defensible number — it survives author-prior
    leakage, including leakage that arrived during pretraining."""
    from src.eval import metrics

    a = scored_fixture("base_few", [8] * 20, [2] * 20, [7] * 20)
    b = scored_fixture("lora_r8", [8] * 20, [2] * 20, [1] * 20)
    standard = metrics.h3_grounding_gap(a + b, arms=["lora_r8"],
                                        condition="mismatched_random")[0]
    strict = metrics.h3_grounding_gap(a + b, arms=["lora_r8"],
                                      condition="mismatched_same_author")[0]
    # Identical under the standard control, separated under the strict one.
    assert standard.detail["mean_arm"] == standard.detail["mean_baseline"]
    assert strict.detail["mean_arm"] > strict.detail["mean_baseline"]


def test_h3_refuses_to_mix_judges():
    """Pooling turns a robustness check into a composite nobody can interpret."""
    from src.eval import metrics

    mixed = (scored_fixture("base_few", [5] * 5, [3] * 5, [3] * 5, judge="a")
             + scored_fixture("base_few", [5] * 5, [3] * 5, [3] * 5, judge="b")
             + scored_fixture("lora_r8", [8] * 5, [2] * 5, [2] * 5, judge="a"))
    try:
        metrics.h3_grounding_gap(mixed, arms=["lora_r8"])
    except AssertionError as error:
        # Asserted on the JUDGE NAMES, not on the wording. Two modules define
        # assert_single_judge with different messages, and a test pinned to one
        # of them passes or fails on prose rather than on behaviour.
        assert "a" in str(error) and "b" in str(error)
        return
    raise AssertionError("records from two judges were pooled")


def test_h4_negative_rho_means_the_measures_agree():
    """Lower perplexity is better and higher judge score is better, so
    agreement is NEGATIVE correlation. Reading the sign backwards inverts the
    conclusion, which is why the direction is recorded on the result."""
    from src.eval import metrics

    ppl = {"a": 10.0, "b": 8.0, "c": 5.0, "d": 4.0}
    agree = {"a": 2.0, "b": 3.0, "c": 6.0, "d": 7.0}
    r = metrics.h4_perplexity_vs_judge(ppl, agree)
    assert r.effect == -1.0
    assert "negative r means the two agree" in r.detail["agreement_sign"]


def test_h4_detects_disagreeing_rankings():
    from src.eval import metrics

    ppl = {"a": 10.0, "b": 8.0, "c": 5.0, "d": 4.0}
    disagree = {"a": 7.0, "b": 6.0, "c": 3.0, "d": 2.0}
    assert metrics.h4_perplexity_vs_judge(ppl, disagree).effect == 1.0


def test_h4_uses_only_arms_with_both_measurements():
    """`template` runs no model, so it HAS no perplexity. That is a property of
    the arm, not a missing measurement."""
    from src.eval import metrics

    r = metrics.h4_perplexity_vs_judge(
        {"a": 10.0, "b": 8.0, "c": 5.0},
        {"a": 2.0, "b": 3.0, "c": 6.0, "template": 1.0})
    assert r.detail["arms"] == ["a", "b", "c"]
    assert r.n == 3


def test_h4_refuses_too_few_arms():
    """Spearman on two points is always +-1 and means nothing."""
    from src.eval import metrics

    try:
        metrics.h4_perplexity_vs_judge({"a": 1.0, "b": 2.0},
                                       {"a": 1.0, "b": 2.0})
    except AssertionError as error:
        assert "at least three arms" in str(error)
        return
    raise AssertionError("Spearman was computed on two arms")


def test_adjust_refuses_a_mixed_judge_family():
    """Holm ranks every p-value against every other, so a family spanning two
    judges makes the primary's threshold depend on the secondary's scores.

    Not hypothetical: a notebook loop rebound the H1/H2 list to the secondary
    judge's results, and the 'primary judge' table silently became a mixed one
    and dropped H1 and H2. Every number in it was real, and nothing raised."""
    from src.eval import metrics

    mixed = [metrics.Result(name="a", judge="gpt4o_mini", p_value=0.01),
             metrics.Result(name="b", judge="gemini_flash", p_value=0.02)]
    try:
        metrics.adjust(mixed)
    except AssertionError as error:
        assert "gpt4o_mini" in str(error) and "gemini_flash" in str(error)
        return
    raise AssertionError("a correction family spanned two judges")


def test_adjust_allows_mechanical_tests_beside_one_judge():
    """H1 and H2 are substring checks with no judge, so they belong in a family
    with either one."""
    from src.eval import metrics

    family = [metrics.Result(name="H1", judge="", p_value=0.001),
              metrics.Result(name="H3", judge="gpt4o_mini", p_value=0.02)]
    out = metrics.adjust(family)
    assert all("p_adjusted" in r.detail for r in out)
