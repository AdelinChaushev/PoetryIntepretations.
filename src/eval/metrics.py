"""Hypothesis tests, effect sizes and confidence intervals.

**Every test here is pre-registered.** The prediction, the statistic and the
judge that decides it were fixed in ``00_introduction.ipynb`` before any result
existed. This module implements them; it does not choose them.

Three rules are enforced rather than remembered, because each is a way an
honest-looking number goes wrong:

*Effect sizes and intervals accompany every p-value.* A significant difference
of 0.4 points on a ten-point scale is a different claim from a significant
difference of 4, and a p-value alone cannot tell them apart.

*A null is "no detected difference", never proof of equality.* H2 and H3 predict
nulls, which makes this load-bearing rather than decorative: failing to detect
an effect at n=152 is not evidence that none exists. :func:`verdict` will not
produce the word "equal".

*Judges are never pooled.* The primary decides every hypothesis; the secondary
re-runs it as a robustness check and is reported beside it. Averaging them would
turn a disagreement — itself a finding about how much a single-judge evaluation
is worth — into a number nobody can interpret.

The statistics come from ``scipy`` and ``statsmodels``. Hand-rolling a bootstrap
or a z-test is how a subtly wrong p-value reaches a report with nothing to catch
it; a `for` loop over pre-registered configurations is not the same kind of
risk.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import config

log = logging.getLogger(__name__)


@dataclass
class Result:
    """One test: what was compared, what came out, and how certain it is."""

    name: str
    #: Which judge decided this. None for model-free metrics.
    judge: str | None = None
    statistic: float = float("nan")
    p_value: float = float("nan")
    #: On the scale of the thing measured — a proportion difference, a rate
    #: ratio, a correlation — never a standardised score alone.
    effect: float = float("nan")
    effect_name: str = ""
    ci: tuple[float, float] = (float("nan"), float("nan"))
    n: int = 0
    detail: dict = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        return self.p_value == self.p_value and self.p_value < config.ALPHA

    def verdict(self) -> str:
        """Plain language, and deliberately unable to claim equality."""
        if self.p_value != self.p_value:
            return "not computable"
        if self.significant:
            return f"detected difference (p={self.p_value:.4f})"
        return f"no detected difference (p={self.p_value:.4f})"

    def line(self) -> str:
        judge = f" [{self.judge}]" if self.judge else ""
        return (f"{self.name}{judge}: {self.effect_name} {self.effect:+.4f} "
                f"95% CI [{self.ci[0]:+.4f}, {self.ci[1]:+.4f}], n={self.n} — "
                f"{self.verdict()}")


# --- proportions --------------------------------------------------------------

def two_proportion_test(successes_a: int, n_a: int,
                        successes_b: int, n_b: int,
                        name: str = "two-proportion") -> Result:
    """Two-proportion z-test, with the difference in proportions as the effect.

    H1 and H2 both use this — same arms, same poems, same style of test; only
    the outcome variable changes. That symmetry is the point: if format
    compliance rises and grounding does not, the model learned the shape of the
    task rather than the task.

    The effect is the raw difference in rates, not an odds ratio, because a
    reader can act on "format compliance rose 22 points" and cannot act on
    "the odds ratio was 3.1".
    """
    from statsmodels.stats.proportion import (confint_proportions_2indep,
                                              proportions_ztest)

    assert n_a > 0 and n_b > 0, "cannot compare proportions with an empty arm"

    statistic, p_value = proportions_ztest([successes_a, successes_b],
                                           [n_a, n_b])
    low, high = confint_proportions_2indep(successes_a, n_a, successes_b, n_b,
                                           method="wald", compare="diff",
                                           alpha=1 - config.CI_LEVEL)
    return Result(
        name=name, statistic=float(statistic), p_value=float(p_value),
        effect=successes_a / n_a - successes_b / n_b,
        effect_name="difference in rate", ci=(float(low), float(high)),
        n=n_a + n_b,
        detail={"rate_a": successes_a / n_a, "rate_b": successes_b / n_b,
                "successes_a": successes_a, "n_a": n_a,
                "successes_b": successes_b, "n_b": n_b},
    )


def binomial_ci(successes: int, n: int) -> tuple[float, float]:
    """Confidence interval for one proportion.

    Wilson rather than normal-approximation: win rates near 0 or 1 are exactly
    where a trivial baseline like ``template`` might land, and the normal
    interval puts its bound outside [0, 1] there.
    """
    from statsmodels.stats.proportion import proportion_confint

    if n == 0:
        return (float("nan"), float("nan"))
    low, high = proportion_confint(successes, n, alpha=1 - config.CI_LEVEL,
                                   method="wilson")
    return (float(low), float(high))


# --- paired per-poem comparisons ----------------------------------------------

def paired_bootstrap_ci(differences: list[float]) -> tuple[float, float]:
    """Bootstrap interval over **paired** per-poem differences.

    Paired because both arms interpreted the same poems: resampling arms
    independently would discard that pairing and overstate the uncertainty. The
    152 test poems form one sample, and the difference for each poem is one
    observation.
    """
    import numpy as np
    from scipy import stats

    values = [d for d in differences if d == d]
    if len(values) < 2:
        return (float("nan"), float("nan"))

    result = stats.bootstrap(
        (np.asarray(values),), np.mean,
        confidence_level=config.CI_LEVEL,
        n_resamples=config.BOOTSTRAP_RESAMPLES,
        random_state=config.SEED, method="percentile",
    )
    return (float(result.confidence_interval.low),
            float(result.confidence_interval.high))


def mann_whitney(a: list[float], b: list[float],
                 name: str = "Mann-Whitney U") -> Result:
    """Mann-Whitney U on two distributions, with a rank-biserial effect size.

    H3 uses this on per-poem grounding gaps. Non-parametric because judge scores
    are ordinal — the distance from 3 to 4 is not the distance from 8 to 9 — and
    a t-test would assume otherwise.

    Rank-biserial correlation is the effect size, reported alongside because a
    U statistic is uninterpretable on its own.
    """
    from scipy import stats

    a = [x for x in a if x == x]
    b = [x for x in b if x == x]
    if not a or not b:
        return Result(name=name, n=len(a) + len(b))

    statistic, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    # Rank-biserial: how often a value from a exceeds one from b, rescaled to
    # [-1, 1]. 0 means the distributions are interleaved.
    effect = 2 * statistic / (len(a) * len(b)) - 1

    return Result(
        name=name, statistic=float(statistic), p_value=float(p_value),
        effect=float(effect), effect_name="rank-biserial",
        ci=paired_bootstrap_ci([x - y for x, y in zip(a, b)])
        if len(a) == len(b) else (float("nan"), float("nan")),
        n=len(a) + len(b),
        detail={"median_a": float(sorted(a)[len(a) // 2]),
                "median_b": float(sorted(b)[len(b) // 2])},
    )


def rank_correlation(x: list[float], y: list[float],
                     name: str = "Spearman") -> Result:
    """Spearman correlation between two rankings.

    H4 compares how the judge ranks the arms against how perplexity ranks them.
    Rank-based because the question is about *ordering*, not about whether a
    perplexity of 5.2 predicts a score of 8.1.

    With one value per arm this runs on very few points. The correlation is
    reported with its n, and a correlation over four or five arms should be read
    as a direction rather than an estimate.
    """
    from scipy import stats

    pairs = [(a, b) for a, b in zip(x, y) if a == a and b == b]
    if len(pairs) < 3:
        return Result(name=name, n=len(pairs))

    xs, ys = zip(*pairs)
    rho, p_value = stats.spearmanr(xs, ys)
    return Result(
        name=name, statistic=float(rho), p_value=float(p_value),
        effect=float(rho), effect_name="rho",
        ci=(float("nan"), float("nan")), n=len(pairs),
        detail={"note": "n is the number of ARMS, not poems; read as a "
                        "direction rather than an estimate"},
    )


# --- agreement and bias -------------------------------------------------------

def cohens_kappa(a: list, b: list, name: str = "inter-judge agreement") -> Result:
    """Cohen's kappa between two judges on identical pairs.

    Raw agreement alone is inflated whenever one verdict dominates — two judges
    that both say "9" to almost everything agree 90% of the time while carrying
    no information. Kappa discounts the agreement chance would produce anyway.

    ``sklearn.metrics.cohen_kappa_score`` rather than a hand-rolled version: the
    chance-correction term is exactly the part that is easy to get subtly wrong
    and impossible to notice.
    """
    from sklearn.metrics import cohen_kappa_score

    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return Result(name=name, n=len(pairs))

    xs, ys = zip(*pairs)
    raw = sum(x == y for x, y in pairs) / len(pairs)

    # Undefined when neither judge varies: every pair agrees, and there is no
    # chance level to correct against. NaN says so; 1.0 would claim perfect
    # agreement on no evidence.
    if len(set(xs)) == 1 and len(set(ys)) == 1:
        kappa = float("nan")
    else:
        kappa = float(cohen_kappa_score(xs, ys))

    return Result(name=name, statistic=kappa, effect=kappa,
                  effect_name="Cohen's kappa", n=len(pairs),
                  ci=binomial_ci(sum(x == y for x, y in pairs), len(pairs)),
                  detail={"raw_agreement": raw,
                          "note": "the CI is on raw agreement, not on kappa"})


def flip_rate(first: list, second: list, name: str = "position bias") -> Result:
    """How often swapping the presentation order flips the verdict.

    Compared against MT-Bench's reported GPT-4 consistency of 65.0%. A judge
    that flips often is measuring position as much as quality, which bounds what
    any single-ordering win rate is worth.

    The interval is binomial on the flip count, since each pair is one Bernoulli
    trial.
    """
    pairs = [(x, y) for x, y in zip(first, second)
             if x is not None and y is not None]
    if not pairs:
        return Result(name=name)

    flips = sum(x != y for x, y in pairs)
    return Result(name=name, effect=flips / len(pairs),
                  effect_name="flip rate", n=len(pairs),
                  ci=binomial_ci(flips, len(pairs)),
                  detail={"flips": flips, "consistent": len(pairs) - flips,
                          "consistency": 1 - flips / len(pairs)})


def win_rate(wins: int, total: int, name: str = "win rate") -> Result:
    """Pairwise win rate against ``base_few``, with a Wilson interval.

    Tested against 0.5 with an exact binomial test rather than a normal
    approximation, because a trivial arm like ``template`` can land near 0 or 1
    where the approximation misbehaves.
    """
    from scipy import stats

    if total == 0:
        return Result(name=name)

    test = stats.binomtest(wins, total, p=0.5, alternative="two-sided")
    return Result(name=name, statistic=float(wins), p_value=float(test.pvalue),
                  effect=wins / total - 0.5, effect_name="advantage over 0.5",
                  ci=binomial_ci(wins, total), n=total,
                  detail={"wins": wins, "losses": total - wins,
                          "rate": wins / total})


# --- reporting ----------------------------------------------------------------

def adjust(results: list[Result]) -> list[Result]:
    """Holm-corrected p-values, attached as detail rather than substituted.

    Reported as a robustness check, never as the decision. H1-H4 are separate
    pre-registered predictions rather than a family screened for any significant
    result — and two of them predict nulls, where correction makes the predicted
    outcome *easier* to obtain and so quietly favours the hypothesis.
    """
    from statsmodels.stats.multitest import multipletests

    usable = [r for r in results if r.p_value == r.p_value]
    if not usable:
        return results

    _, adjusted, _, _ = multipletests(
        [r.p_value for r in usable], alpha=config.ALPHA,
        method=config.MULTIPLE_COMPARISON_METHOD)
    for result, value in zip(usable, adjusted):
        result.detail["p_adjusted"] = float(value)
        result.detail["adjustment"] = config.MULTIPLE_COMPARISON_METHOD
    return results


def to_frame(results: list[Result]):
    """One row per test, for the report. Generated, never hand-typed."""
    import pandas as pd

    return pd.DataFrame([{
        "test": r.name,
        "judge": r.judge or "",
        "n": r.n,
        "effect": r.effect,
        "effect_name": r.effect_name,
        "ci_low": r.ci[0],
        "ci_high": r.ci[1],
        "p_value": r.p_value,
        "p_adjusted": r.detail.get("p_adjusted", float("nan")),
        "verdict": r.verdict(),
    } for r in results])


def assert_single_judge(results: list[Result]) -> None:
    """Refuse to aggregate results decided by different judges.

    Pooling is the failure that turns a robustness check into a composite metric
    nobody can interpret. The primary decides; the secondary is reported beside
    it, and where they disagree that disagreement is itself the finding.
    """
    judges = {r.judge for r in results if r.judge}
    assert len(judges) <= 1, (
        f"results from {sorted(judges)} cannot be aggregated. Report the "
        f"primary judge's number as the result and the secondary beside it — "
        f"averaging them hides a disagreement that is itself a finding.")
