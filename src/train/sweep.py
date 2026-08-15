"""Drive the hyperparameter sweep and record every run.

Two stages, and they answer different kinds of question.

**Stage 1 — tuning.** Rank and learning rate as a full product, because they
interact: ``alpha/r`` scaling and learning rate both control how far the adapter
moves per step, so the best rank at one learning rate need not be the best at
another. Nine runs.

**Stage 2 — reported curves, at the winning config.** Data size and masking are
not hyperparameters. The final runs use all the data regardless, and masking is
never in doubt. They are here because "does more data improve grounding, or only
format?" is one of the fine-tuning questions the project asks, and because
training unmasked is a cheap sanity check on the highest-risk code in the
pipeline.

Running them *at the winner* rather than at the defaults matters: a curve drawn
at a configuration already rejected describes a model nobody will train.

**The selection metric is fixed in config before any run.** With nine runs there
is always some metric under which some config looks best, and choosing it
afterwards is the freedom the pre-registration exists to remove. Judge scores
never select — they are the outcome, and selecting on them would fit
hyperparameters to the thing being reported.
"""

from __future__ import annotations

import logging

import config

log = logging.getLogger(__name__)


def tuning_grid() -> list[dict]:
    """The rank x learning-rate product. A grid, not one-at-a-time."""
    return [{"rank": rank, "learning_rate": lr}
            for rank in config.RANK_SWEEP
            for lr in config.LR_SWEEP]


def reported_curves(winner: dict) -> list[dict]:
    """Data-size and masking runs, at the winning configuration.

    The winner's own data-size point (the full corpus) and masking setting
    (masked) already exist from stage 1, so they are not repeated.

    Early stopping is not set here: it is on globally, and stating it per-axis
    would invite the two to drift apart.
    """
    runs = []
    for size in config.DATA_SIZE_SWEEP:
        if size is not None:
            runs.append({**winner, "data_size": size})
    for masking in config.MASKING_SWEEP:
        if masking != "masked":
            runs.append({**winner, "masking": masking})
    return runs


def run_name(spec: dict) -> str:
    """A name that encodes what varied, so runs.csv reads without a decoder."""
    parts = [f"r{spec.get('rank', config.LORA_RANK)}",
             f"lr{spec.get('learning_rate', config.LEARNING_RATE):g}"]
    if spec.get("data_size") is not None:
        parts.append(f"n{spec['data_size']}")
    if spec.get("masking", "masked") != "masked":
        parts.append(spec["masking"])
    return "sweep_" + "_".join(parts)


def select_winner(records: list[dict]) -> dict:
    """The best run by the pre-registered metric.

    Ties break toward FEWER trainable parameters, so a config matching another's
    loss with less capacity wins — the direction Hu et al.'s saturation finding
    predicts, which keeps the tiebreak from quietly favouring the outcome.
    """
    metric = config.SWEEP_SELECTION_METRIC

    # Rows from a different model must never compete. A five-step gpt2 smoke run
    # can post a lower validation loss than a real one — fewer examples, a
    # smaller vocabulary — and would then be returned as the winning
    # configuration despite never having been trained at this scale. The
    # selection is pre-registered on `metric` alone, so nothing else in the
    # function will notice.
    def eligible(record: dict) -> bool:
        if record.get(metric) is None or record[metric] != record[metric]:
            return False                            # None or NaN
        if str(record.get("smoke", "")).lower() in ("true", "1"):
            return False
        return record.get("model", config.MODEL) == config.MODEL

    rejected = [r["run"] for r in records if not eligible(r)]
    if rejected:
        log.warning("excluded %d run(s) from selection (smoke, wrong model, or "
                    "no %s): %s", len(rejected), metric, ", ".join(rejected))

    usable = [r for r in records if eligible(r)]
    assert usable, (
        f"no eligible run recorded {metric} for {config.MODEL}; nothing to "
        f"select on. Rejected: {rejected}")

    sign = 1 if config.SWEEP_SELECTION_LOWER_IS_BETTER else -1
    best = min(usable, key=lambda r: (sign * r[metric], r["trainable_params"]))
    log.info("winner: %s with %s=%.4f (%s trainable)", best["run"], metric,
             best[metric], f"{best['trainable_params']:,}")
    return {"rank": best["rank"], "learning_rate": best["learning_rate"]}


def run_one(spec: dict, pairs: list[dict], fold: int | None = None):
    """Train one sweep configuration and return its ``runs.csv`` record.

    Everything is rebuilt per run — model, adapters, optimiser — because an
    adapter carried between runs would make each one a continuation of the last
    rather than an independent configuration.
    """
    from src.data import splits
    from src.model import setup
    from src.train import dataset, loop

    fold = config.SINGLE_SPLIT_FOLD if fold is None else fold
    rank = spec.get("rank", config.LORA_RANK)
    setup.assert_matched_learning_rate(
        "lora", spec.get("learning_rate", config.LEARNING_RATE))

    # NOT an inline p.get("fold_id") comparison: the pairs file carries no
    # fold_id, so that test is true for every poem and the held-out fold ends
    # up in training. splits.training_partition asserts the filter bit.
    training = splits.training_partition(pairs, fold)
    if spec.get("data_size"):
        # Sorted by id first, so the subset is deterministic rather than
        # whatever order the corpus happened to arrive in.
        training = sorted(training, key=lambda p: p["poem_id"])[:spec["data_size"]]

    tokenizer = setup.load_tokenizer()
    model = setup.apply_lora(setup.load_base_model(), rank=rank)
    examples = dataset.build_dataset(training, tokenizer)

    return loop.train(
        model, tokenizer, examples, pairs,
        run_name=run_name(spec),
        # None, not False: an absent key must fall through to the config
        # default (on), not silently disable early stopping for every grid run.
        early_stopping=spec.get("early_stopping"),
        rank=rank,
        learning_rate=spec.get("learning_rate", config.LEARNING_RATE),
    )


def plan() -> dict:
    """What the sweep will run, without running it.

    Printed before a GPU session starts, because a sweep whose size is only
    discovered halfway through is a sweep that gets abandoned halfway through.
    """
    grid = tuning_grid()
    curves = reported_curves({"rank": config.LORA_RANK,
                              "learning_rate": config.LEARNING_RATE})
    return {
        "tuning_runs": len(grid),
        "curve_runs": len(curves),
        "total": len(grid) + len(curves),
        "grid": [run_name(s) for s in grid],
        "curves": [run_name(s) for s in curves],
        "selection": f"{config.SWEEP_SELECTION_METRIC}, "
                     f"{'lower' if config.SWEEP_SELECTION_LOWER_IS_BETTER else 'higher'}"
                     f" is better, ties to fewer trainable params",
    }
