"""Drive the hyperparameter sweep and record every run.

Two stages, and they answer different kinds of question.

**Stage 1 — tuning, in two sequential passes.** Learning rate first at the
default rank, then rank at the learning rate that won. Five runs.

Sequential, not independent: sweeping rank around a fixed default would draw the
rank curve at a configuration the search may already have rejected. Greedy
coordinate descent costs the same five runs and searches strictly better.

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


def lr_specs() -> list[dict]:
    """Stage 1a — learning rate, at the default rank.

    Learning rate goes first because ``alpha`` is derived as ``2 * rank``, so
    ``alpha/r`` is constant across the rank sweep and the effective step size is
    rank-independent *by construction*. A rate chosen at one rank therefore
    transfers to the others, which is what makes the sequential design sound
    rather than merely cheap.
    """
    return [{"rank": config.LORA_RANK, "learning_rate": lr}
            for lr in config.LR_SWEEP]


def rank_specs(learning_rate: float) -> list[dict]:
    """Stage 1b — rank, at the learning rate stage 1a chose.

    **Sequential, not independent.** Sweeping rank at a fixed *default* rate
    would draw the rank curve at a configuration the search may already have
    rejected — the same objection that puts the ablations at the winner rather
    than at the defaults. Running it at the chosen rate costs nothing extra:
    ``run_cv_stage`` skips whatever is already recorded.

    This is greedy coordinate descent. It does not find an interaction the way a
    full product would, and that stays in the limitations — but for the same
    five runs it searches strictly better than varying each axis around a fixed
    default.
    """
    return [{"rank": rank, "learning_rate": learning_rate}
            for rank in config.RANK_SWEEP]


def tuning_grid() -> list[dict]:
    """Every tuning run, for planning and cost estimates only.

    The stages run separately — :func:`lr_specs`, then :func:`rank_specs` at the
    winner — because the second depends on the first's result. This flattened
    view exists so :func:`plan` can report the size before any GPU time is
    spent, with the duplicate point counted once.
    """
    specs, seen = [], set()
    for spec in lr_specs() + rank_specs(config.LEARNING_RATE):
        key = (spec["rank"], spec["learning_rate"])
        if key not in seen:
            seen.add(key)
            specs.append(spec)
    return specs


def run_name(spec: dict) -> str:
    """A name that encodes what varied, so runs.csv reads without a decoder."""
    parts = [f"r{spec.get('rank', config.LORA_RANK)}",
             f"lr{spec.get('learning_rate', config.LEARNING_RATE):g}"]
    if spec.get("data_size") is not None:
        parts.append(f"n{spec['data_size']}")
    if spec.get("masking", "masked") != "masked":
        parts.append(spec["masking"])
    return "sweep_" + "_".join(parts)


def assert_stage_complete(specs: list[dict], stage: str = "stage") -> None:
    """Every run in a stage must be recorded before its result is used.

    **The sequential design has a dependency that nothing else enforces.**
    Stage 1b sweeps rank *at the rate stage 1a chose*, so selecting from a
    partial 1a and proceeding runs the whole rank sweep at a rate that may not
    survive the missing runs — and if it does not, every 1b result is at the
    wrong configuration and must be repeated.

    This happened: ``max_runs=2`` stopped 1a at two of three learning rates, the
    notebook selected a winner from those two, and 1b ran to completion at it.
    Nothing raised, because a partial stage and a complete one look identical to
    ``select_winner`` — it minimises over whatever it is handed.
    """
    done = load_completed()
    missing = [run_name(s) for s in specs if not already_done(s, done)]
    assert not missing, (
        f"{stage} is incomplete: {missing} not recorded. Selecting now would "
        f"choose from a partial sweep, and anything run at that choice would "
        f"have to be repeated if a missing run turns out to win. Finish the "
        f"stage first — run_cv_stage skips what is already done.")


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


# --- cross-validated tuning ---------------------------------------------------
#
# CV belongs here and nowhere else. Its job is to stop a hyperparameter choice
# being hostage to one slice of authors. It is NOT a way to define held-out
# data: applied to the test partition it produces k models where one is wanted,
# and LoRA adapters trained on different data cannot be merged into a single
# artefact.

def cv_run_name(spec: dict, fold: int) -> str:
    """Name for one (configuration, fold) tuning run."""
    return f"cv_r{spec['rank']}_lr{spec['learning_rate']:g}_f{fold}"


def cv_specs(k: int | None = None) -> list[dict]:
    """Every (configuration, fold) pair the tuning stage will run.

    Configurations vary one axis at a time — rank at the default rate, then rate
    at the default rank, with the shared point counted once. The product would
    be more thorough and costs k runs per cell rather than k per config, which
    at 1.3 hours a run does not fit a GPU week.
    """
    folds = config.TUNING_FOLDS if k is None else k
    return [{**spec, "fold": fold, "run": cv_run_name(spec, fold)}
            for spec in tuning_grid()
            for fold in range(folds)]


def select_winner_cv(records: list[dict]) -> dict:
    """The configuration with the best **mean** validation loss across folds.

    Averaging is the entire point of cross-validating a hyperparameter choice.
    Taking the single best fold would select the luckiest slice of authors —
    exactly the fragility CV exists to remove — and with three folds per config
    the best-of-three is a noticeably optimistic estimate.

    A configuration missing any of its folds is excluded rather than averaged
    over what happens to be present, since a mean of two is not comparable with
    a mean of three.
    """
    metric = config.SWEEP_SELECTION_METRIC
    assert "heldout" not in metric, (
        f"SWEEP_SELECTION_METRIC is {metric!r}; selecting on held-out data "
        f"turns the test set into a validation set")

    grouped: dict = {}
    for row in records:
        if row.get(metric) in (None, "", "None"):
            continue
        if str(row.get("smoke", "")).lower() in ("true", "1"):
            continue
        if row.get("model", config.MODEL) != config.MODEL:
            continue
        key = (int(float(row["rank"])), float(row["learning_rate"]))
        grouped.setdefault(key, []).append(float(row[metric]))

    expected = config.TUNING_FOLDS
    complete = {k: v for k, v in grouped.items() if len(v) == expected}
    partial = {k: len(v) for k, v in grouped.items() if len(v) != expected}
    if partial:
        log.warning("excluded %d configuration(s) missing folds: %s",
                    len(partial), partial)

    assert complete, (
        f"no configuration has all {expected} folds recorded; "
        f"found {[(k, len(v)) for k, v in grouped.items()]}")

    sign = 1 if config.SWEEP_SELECTION_LOWER_IS_BETTER else -1
    means = {k: sum(v) / len(v) for k, v in complete.items()}
    # Ties break toward the smaller rank, the direction Hu et al.'s saturation
    # finding predicts, so the tiebreak cannot quietly favour the outcome.
    best = min(means, key=lambda k: (sign * means[k], k[0]))

    log.info("CV winner: rank %d, lr %g — mean %s %.4f over %d folds",
             best[0], best[1], metric, means[best], expected)
    for key in sorted(means, key=lambda k: sign * means[k]):
        log.info("    r%-3d lr%-8g mean %.4f  (folds: %s)", key[0], key[1],
                 means[key], ", ".join(f"{v:.4f}" for v in complete[key]))
    return {"rank": best[0], "learning_rate": best[1]}


def final_specs(winner: dict) -> list[dict]:
    """The two LoRA arms the results are reported from.

    Each is trained on the whole pool minus a validation slice, then measured
    **once** on the test set. Named ``lora_r{rank}`` with no fold suffix,
    because there is exactly one adapter per arm — which is what the README's
    load snippet promises and what the fold design could not provide.

    **The ranks come from the pre-registration, not from the sweep.** They name
    the arms; letting a winning rank of 4 rename them would change the
    experiment after seeing results. The sweep supplies the learning rate, and
    the rank question is answered by the CV curve over {4, 8, 16}.

    Both arms see the same pool and the same validation slice, so ``lora_r8`` vs
    ``lora_r16`` isolates adapter capacity and nothing else — which the 5-fold
    version could not do, since it compared one fold's r8 against one fold's r16
    and the difference also carried which poems each had seen.
    """
    lr = float(winner.get("learning_rate", config.LEARNING_RATE))
    return [{"rank": rank, "learning_rate": lr, "run": f"lora_r{rank}"}
            for rank in config.LORA_ARM_RANKS]


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


def run_cv_fold(spec: dict, pairs: list[dict], save_adapter: bool = True):
    """One tuning run: train on the other tuning folds, score on this one.

    Everything is rebuilt per run — model, adapters, optimiser — because an
    adapter carried between runs would make each a continuation of the last
    rather than an independent configuration.
    """
    from src.data import splits
    from src.model import setup
    from src.train import dataset, loop

    fold = spec["fold"]
    rank = int(spec["rank"])
    setup.assert_matched_learning_rate("lora", spec["learning_rate"])

    train_pairs, held_pairs = splits.tuning_partition(pairs, fold)

    tokenizer = setup.load_tokenizer()
    model = setup.apply_lora(setup.load_base_model(), rank=rank)

    return loop.train(
        model, tokenizer, dataset.build_dataset(train_pairs, tokenizer), pairs,
        run_name=spec["run"],
        fold=fold,
        save_adapter=save_adapter,
        # This fold's held-out poems, not the test set. Tuning never touches the
        # test set — splits.assert_tuning_never_sees_test checked that before
        # the split was written to disk.
        heldout=dataset.build_dataset(held_pairs, tokenizer),
        early_stopping=spec.get("early_stopping"),
        rank=rank,
        learning_rate=spec["learning_rate"],
        masking=spec.get("masking", "masked"),
    )


def run_final(spec: dict, pairs: list[dict], save_adapter: bool = True):
    """One of the two LoRA arms the results are reported from.

    Trains on the pool minus a validation slice, and is measured **once** on the
    test set — the only time any model sees it.

    The validation slice is drawn from authors the tuning stage never used, so
    the stopping point is not selected on data whose hyperparameters were also
    selected on it.
    """
    from src.data import splits
    from src.model import setup
    from src.train import dataset, loop

    rank = int(spec["rank"])
    setup.assert_matched_learning_rate("lora", spec["learning_rate"])

    pool = splits.pool_partition(pairs)
    test = splits.test_partition(pairs)
    assert not ({p["poem_id"] for p in pool} & {p["poem_id"] for p in test}), (
        "the pool and the test set overlap — this run would train on the data "
        "it is about to be measured by")

    tokenizer = setup.load_tokenizer()
    model = setup.apply_lora(setup.load_base_model(), rank=rank)

    return loop.train(
        model, tokenizer, dataset.build_dataset(pool, tokenizer), pairs,
        run_name=spec["run"],
        save_adapter=save_adapter,
        heldout=dataset.build_dataset(test, tokenizer),
        prefer_unused=splits.untouched_authors(pairs),
        early_stopping=spec.get("early_stopping"),
        rank=rank,
        learning_rate=spec["learning_rate"],
        masking=spec.get("masking", "masked"),
    )


# --- ablations ----------------------------------------------------------------
#
# Neither of these feeds H1-H4. They are reportable findings in their own right,
# and both are measured on the SAME test set as the final model so the numbers
# sit on one scale.

def ablation_specs(winner: dict) -> list[dict]:
    """Data-size and masking runs, at the winning configuration.

    **Data size** puts points either side of LIMA's 1,000-example threshold, so
    the comparison is where our curve flattens relative to theirs rather than
    merely that we also used few examples. The full-corpus point is the
    ``lora_r8`` arm itself and is not repeated.

    **Masking** is the only empirical check that ``completion_only_loss`` does
    what the project assumes. H1, H2 and H3 all rest on the model being scored
    on the interpretation and not on the poem; the tests verify the flag reaches
    TRL, and this verifies it changes the result.

    Both run at the tuned learning rate rather than the default: a curve drawn
    at a configuration the search rejected describes a model nobody will train.
    """
    # At the PRIMARY arm, not at the winning rank. The curve's right-hand end
    # has to be a run that already exists, and the full-pool run at this rank is
    # `lora_r8` — the arm H1-H3 are stated about. Drawing it at a winning rank
    # of 4 would leave the n=full point belonging to no reported model.
    rank = config.PRIMARY_LORA_RANK
    lr = float(winner["learning_rate"])
    specs = [{"rank": rank, "learning_rate": lr, "data_size": size,
              "run": f"ablation_n{size}"}
             for size in config.DATA_SIZE_SWEEP if size is not None]
    specs += [{"rank": rank, "learning_rate": lr, "masking": masking,
               "run": f"ablation_{masking}"}
              for masking in config.MASKING_SWEEP if masking != "masked"]
    return specs


def run_ablation(spec: dict, pairs: list[dict], save_adapter: bool = True):
    """One ablation run, trained on the pool and measured on the test set.

    The test set is the same one the final model is measured on, so a data-size
    curve and the final number sit on one scale. Nothing here touches it during
    training — ``pool_partition`` excludes it, and ``run_final`` asserts the two
    do not overlap.
    """
    from src.data import splits
    from src.model import setup
    from src.train import dataset, loop

    rank = int(spec["rank"])
    setup.assert_matched_learning_rate("lora", spec["learning_rate"])

    pool = splits.pool_partition(pairs)
    if spec.get("data_size"):
        # Sorted by id first, so the subset is deterministic rather than
        # whatever order the corpus arrived in. Authors are NOT kept whole here:
        # the question is how much data helps, and holding authors together
        # would confound size with author coverage.
        pool = sorted(pool, key=lambda p: p["poem_id"])[:spec["data_size"]]
        log.info("ablation on %d poems", len(pool))

    test = splits.test_partition(pairs)
    assert not ({p["poem_id"] for p in pool} & {p["poem_id"] for p in test}), (
        "the training subset overlaps the test set")

    tokenizer = setup.load_tokenizer()
    model = setup.apply_lora(setup.load_base_model(), rank=rank)

    return loop.train(
        model, tokenizer, dataset.build_dataset(pool, tokenizer), pairs,
        run_name=spec["run"],
        save_adapter=save_adapter,
        heldout=dataset.build_dataset(test, tokenizer),
        prefer_unused=splits.untouched_authors(pairs),
        early_stopping=spec.get("early_stopping"),
        rank=rank,
        learning_rate=spec["learning_rate"],
        # Threaded through, or the unmasked run trains masked and the axis
        # reports a comparison that never happened.
        masking=spec.get("masking", "masked"),
    )


def run_ablation_stage(pairs: list[dict], winner: dict,
                       max_runs: int | None = None) -> list[dict]:
    """Run every ablation not already recorded."""
    import csv
    from pathlib import Path

    specs = ablation_specs(winner)
    done = load_completed()
    pending = [s for s in specs if s["run"] not in done]
    recorded = len(specs) - len(pending)

    if max_runs is not None and len(pending) > max_runs:
        log.warning("%d pending; doing %d this batch.", len(pending), max_runs)
        pending = pending[:max_runs]

    log.info("ablations: %d run(s), %d already recorded, %d still to run, "
             "%d in this batch", len(specs), recorded, len(specs) - recorded,
             len(pending))

    for index, spec in enumerate(pending, 1):
        log.info("[ablation %d/%d] %s", index, len(pending), spec["run"])
        run_ablation(spec, pairs)

    path = Path(config.RUNS_CSV_PATH)
    if not path.exists():
        return []
    wanted = {s["run"] for s in specs}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["run"] in wanted]
    return list({r["run"]: r for r in rows}.values())


def run_cv_stage(pairs: list[dict], max_runs: int | None = None) -> list[dict]:
    """Run every (configuration, fold) tuning run not already recorded."""
    import csv
    from pathlib import Path

    specs = cv_specs()
    done = load_completed()
    pending = [s for s in specs if s["run"] not in done]
    recorded = len(specs) - len(pending)

    if max_runs is not None and len(pending) > max_runs:
        log.warning("%d pending; doing %d this batch. Archive %s before the "
                    "next one — it does not survive the session.",
                    len(pending), max_runs, config.RUNS_CSV_PATH.name)
        pending = pending[:max_runs]

    log.info("tuning: %d run(s), %d already recorded, %d still to run, "
             "%d in this batch", len(specs), recorded, len(specs) - recorded,
             len(pending))

    for index, spec in enumerate(pending, 1):
        log.info("[tuning %d/%d] %s", index, len(pending), spec["run"])
        run_cv_fold(spec, pairs)

    path = Path(config.RUNS_CSV_PATH)
    if not path.exists():
        return []
    wanted = {s["run"] for s in specs}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["run"] in wanted]
    return list({r["run"]: r for r in rows}.values())


def plan(winner: dict | None = None) -> dict:
    """What the whole GPU programme will run, without running it.

    Printed before a session starts, because a sweep whose size is only
    discovered halfway through is a sweep that gets abandoned halfway through.

    ``winner`` is only needed for the ablation names; the *count* is the same
    whichever configuration wins, so the defaults stand in when planning ahead
    of the tuning stage.
    """
    winner = winner or {"rank": config.LORA_RANK,
                        "learning_rate": config.LEARNING_RATE}
    tuning = cv_specs()
    final = final_specs(winner)
    ablations = ablation_specs(winner)
    return {
        "tuning_runs": len(tuning),
        "final_runs": len(final),
        "ablation_runs": len(ablations),
        "total": len(tuning) + len(final) + len(ablations),
        "tuning": [s["run"] for s in tuning],
        "final": [s["run"] for s in final],
        "ablations": [s["run"] for s in ablations],
        "selection": f"{config.SWEEP_SELECTION_METRIC}, "
                     f"{'lower' if config.SWEEP_SELECTION_LOWER_IS_BETTER else 'higher'}"
                     f" is better, averaged over {config.TUNING_FOLDS} folds",
    }
