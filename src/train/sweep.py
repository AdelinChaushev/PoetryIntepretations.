"""Drive the hyperparameter sweep and record every run.

Two stages, and they answer different kinds of question.

**Stage 1 — tuning.** Rank and learning rate varied **one at a time**, with the
other at its default. Five runs.

An earlier version ran the 3x3 product, arguing that the two interact through
``alpha/r`` scaling. They do — but at ~1.3 hours per full-corpus run that is
11.6 GPU-hours for tuning alone, and a sweep that cannot finish inside a
12-hour Kaggle session measures nothing. One session was lost that way. The
interaction goes in the limitations as unmeasured rather than in the budget as
unaffordable.

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
    """Rank and learning rate, varied **one at a time** with the other at its
    default.

    Not the 3x3 product. A previous version ran the full grid on the argument
    that rank and learning rate interact through ``alpha/r`` scaling — which is
    true, but it costs nine full-corpus runs at ~1.3 hours each, and a session
    that cannot finish measures nothing at all. CLAUDE.md specifies one-at-a-time
    and this returns to it.

    The interaction is not lost, only unmeasured, and that belongs in the
    limitations: a rank that would have won at a different learning rate is not
    detectable here. The rank axis is reported as a curve against Hu et al.'s
    saturation finding either way, which is what it was mainly for.

    Duplicates are collapsed: the default rank at the default learning rate
    appears on both axes and is trained once.
    """
    specs, seen = [], set()
    for rank in config.RANK_SWEEP:
        specs.append({"rank": rank, "learning_rate": config.LEARNING_RATE})
    for lr in config.LR_SWEEP:
        specs.append({"rank": config.LORA_RANK, "learning_rate": lr})

    unique = []
    for spec in specs:
        key = (spec["rank"], spec["learning_rate"])
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique


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

    # **Selection may never read the held-out fold.** That fold is measured once
    # per run and recorded, which is safe only because nothing is chosen on it.
    # The moment a hyperparameter is picked by held-out score, the held-out fold
    # stops being a test set and becomes a second validation set — and the
    # numbers reported from it become as optimistic as validation already is,
    # which is the exact problem heldout_perplexity was added to fix.
    assert "heldout" not in metric, (
        f"SWEEP_SELECTION_METRIC is {metric!r}. Selecting on the held-out fold "
        f"turns it into a validation set: every number later reported from it "
        f"would carry the selection bias that measuring it separately exists to "
        f"avoid. Select on validation; report on held-out.")

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
    # int, not whatever the CSV produced: peft builds nn.Linear(in_features, r)
    # and a float r fails several frames deep with no mention of the rank.
    return {"rank": int(best["rank"]),
            "learning_rate": float(best["learning_rate"])}


def load_completed() -> dict:
    """Run name -> the row ``runs.csv`` recorded for it.

    **Resumability is not a nicety here.** Nineteen runs will not fit a single
    Kaggle session reliably, and a session that dies at run 14 must not restart
    at run 1 — that is how a sweep gets abandoned.

    Returns the whole row rather than just the name, because a name alone is not
    enough to decide a run is done: see :func:`already_done`.
    """
    import csv
    from pathlib import Path

    path = Path(config.RUNS_CSV_PATH)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["run"]: row for row in csv.DictReader(handle)
                if row.get("run")}


def already_done(spec: dict, completed: dict) -> bool:
    """Whether ``spec`` has been run **at this configuration**.

    Name matching alone is not sufficient, and the failure is concrete rather
    than theoretical. The day-3 pilot is recorded as ``lora_r8_fold0`` at the
    default learning rate — the same name stage 3 gives its fold-0 run. If the
    sweep chooses a different learning rate, skipping on the name would keep the
    pilot's adapter and ship a final result trained at a configuration the sweep
    rejected. Nothing would raise: the row exists, the adapter exists, and the
    name is right.

    So the recorded hyperparameters must match too.
    """
    row = completed.get(spec.get("run") or run_name(spec))
    if row is None:
        return False

    for key, default in (("rank", config.LORA_RANK),
                         ("learning_rate", config.LEARNING_RATE)):
        recorded, wanted = row.get(key), spec.get(key, default)
        if recorded in (None, "", "None"):
            return False
        if abs(float(recorded) - float(wanted)) > 1e-12:
            log.info("%s was run at %s=%s but is now specified as %s; "
                     "re-running", row["run"], key, recorded, wanted)
            return False
    return True


def final_specs(winner: dict) -> list[dict]:
    """The six runs the results are reported from.

    Five ``lora_r8``, one per fold — this is the cross-validation, and it is the
    number 5-fold was bought for: how much the result moves with *which* poems
    were trained on. Plus one ``lora_r16`` on the single-split fold, so
    ``lora_r8`` vs ``lora_r16`` compares fold-1 against fold-1 rather than a
    fold-averaged r8 against a single-split r16.

    **The ranks are fixed at 8 and 16 regardless of what the sweep chose.** They
    name the pre-registered arms; the sweep supplies the learning rate. Letting
    a winning rank of 4 rename the arms would change the experiment after seeing
    results, which is the freedom the pre-registration exists to remove — and
    the rank question is answered by the reported curve, not by the arms.
    """
    lr = winner.get("learning_rate", config.LEARNING_RATE)
    specs = [{"rank": 8, "learning_rate": lr, "fold": fold,
              "run": f"lora_r8_fold{fold}"} for fold in range(config.N_FOLDS)]
    specs.append({"rank": 16, "learning_rate": lr,
                  "fold": config.SINGLE_SPLIT_FOLD,
                  "run": f"lora_r16_fold{config.SINGLE_SPLIT_FOLD}"})
    return specs


def run_stage(specs: list[dict], pairs: list[dict], stage: str,
              save_adapters: bool = False,
              requires: tuple[str, ...] = (),
              max_runs: int | None = None) -> list[dict]:
    """Run every spec not already recorded, and return all records for them.

    Args:
        save_adapters: False for sweep stages. Nine grid points would otherwise
            collide — every rank-8 configuration writes to the same
            ``adapter_dir(8, fold)``, so the last silently overwrites the rest
            and the surviving adapter belongs to no recorded run in particular.
            The final runs are the only ones whose weights are ever loaded.
    """
    done = load_completed()

    # A row can match on name and configuration and still be unusable, because
    # it predates a column the stage needs. The day-3 pilot is exactly that: it
    # is recorded as lora_r8_fold0 at the default learning rate, but ran before
    # heldout_perplexity existed. If the sweep happens to pick that same
    # configuration, skipping it would leave H4 without a value for that fold
    # and nothing would complain.
    def usable(spec: dict) -> bool:
        if not already_done(spec, done):
            return False
        row = done[spec.get("run") or run_name(spec)]
        absent = [f for f in requires if row.get(f) in (None, "", "None")]
        if absent:
            log.info("%s exists but is missing %s; re-running",
                     row["run"], ", ".join(absent))
            return False
        return True

    pending = [s for s in specs if not usable(s)]

    # Batching exists because `runs.csv` lives in /kaggle/working, which is wiped
    # when a session ends. A stage that runs for eight hours and is cut off at
    # the cap loses every row it wrote. Running a few at a time and downloading
    # between batches bounds the loss to the current batch.
    if max_runs is not None and len(pending) > max_runs:
        log.warning("%d run(s) pending; doing %d this batch. DOWNLOAD runs.csv "
                    "before starting the next one — /kaggle/working does not "
                    "survive the session.", len(pending), max_runs)
        pending = pending[:max_runs]
    log.info("%s: %d run(s), %d already recorded, %d to run",
             stage, len(specs), len(specs) - len(pending), len(pending))

    for index, spec in enumerate(pending, 1):
        name = spec.get("run") or run_name(spec)
        log.info("[%s %d/%d] %s", stage, index, len(pending), name)
        run_one(spec, pairs, fold=spec.get("fold"),
                save_adapter=save_adapters)

    import csv
    from pathlib import Path

    path = Path(config.RUNS_CSV_PATH)
    if not path.exists():
        return []
    wanted = {s.get("run") or run_name(s) for s in specs}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["run"] in wanted]
    # Newest per name wins: a re-run at a different configuration appends
    # rather than replacing, so the earlier row must not be returned as well.
    return list({row["run"]: row for row in rows}.values())


#: Columns that must come back as ``int``, not ``float``. CSV round-trips
#: everything as text, and a blanket ``float()`` turns rank 16 into 16.0 — which
#: propagates through select_winner into ``LoraConfig(r=16.0)`` and dies deep
#: inside peft at ``nn.Linear(in_features, 16.0)``, with a message naming
#: neither the rank nor the sweep. It also renames the run ``sweep_r16.0_...``,
#: so the stage would not match its own rows on a resume.
INTEGER_COLUMNS: tuple[str, ...] = ("rank", "trainable_params", "steps_run",
                                    "n_train", "n_validation", "n_heldout",
                                    "fold", "max_steps", "seed")

FLOAT_COLUMNS: tuple[str, ...] = ("final_val_loss", "final_train_loss",
                                  "val_perplexity", "heldout_perplexity",
                                  "heldout_loss", "learning_rate",
                                  "wall_clock_seconds")


def coerce(records: list[dict]) -> list[dict]:
    """Numbers read back from CSV arrive as strings; restore their real types."""
    out = []
    for row in records:
        row = dict(row)
        for key in INTEGER_COLUMNS:
            if row.get(key) not in (None, "", "None"):
                try:
                    row[key] = int(float(row[key]))
                except ValueError:
                    pass
        for key in FLOAT_COLUMNS:
            if row.get(key) not in (None, "", "None"):
                try:
                    row[key] = float(row[key])
                except ValueError:
                    pass
        out.append(row)
    return out


def run_one(spec: dict, pairs: list[dict], fold: int | None = None,
            save_adapter: bool = False):
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
        run_name=spec.get("run") or run_name(spec),
        fold=fold,
        save_adapter=save_adapter,
        # None, not False: an absent key must fall through to the config
        # default (on), not silently disable early stopping for every grid run.
        early_stopping=spec.get("early_stopping"),
        rank=rank,
        learning_rate=spec.get("learning_rate", config.LEARNING_RATE),
        # Threaded through, or the unmasked run trains masked and the axis
        # reports a comparison that never happened.
        masking=spec.get("masking", "masked"),
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
