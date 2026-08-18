"""Tests for the sweep driver.

The selection rule is the part worth pinning. With nine runs there is always
some metric under which some config looks best, so the metric, its direction and
the tiebreak are all fixed in config before anything runs — and judge scores are
never among them.
"""

from __future__ import annotations

import config
from src.train import sweep


def record(name, val_loss, params=737_280, rank=8, lr=2e-4):
    return {"run": name, "final_val_loss": val_loss, "trainable_params": params,
            "rank": rank, "learning_rate": lr}


# --- the grid -----------------------------------------------------------------

def test_tuning_is_one_at_a_time_not_a_product():
    """CLAUDE.md specifies one-at-a-time and says explicitly not to run the full
    grid. A previous version ran the 3x3 product: 11.6 GPU-hours for tuning
    alone, which does not fit a 12-hour Kaggle session — and one session was
    lost to exactly that."""
    grid = sweep.tuning_grid()
    assert len(grid) == len(config.RANK_SWEEP) + len(config.LR_SWEEP) - 1
    assert len({(s["rank"], s["learning_rate"]) for s in grid}) == len(grid)


def test_ablations_run_at_the_tuned_rate_and_the_primary_arm():
    """The rate comes from the sweep — a curve drawn at a rejected rate
    describes a model nobody will train. The RANK does not: the curve's
    right-hand end has to be a run that already exists, and that is lora_r8. A
    winning rank of 4 would leave the n=full point belonging to no arm."""
    for spec in sweep.ablation_specs({"rank": 4, "learning_rate": 5e-4}):
        assert spec["learning_rate"] == 5e-4
        assert spec["rank"] == config.PRIMARY_LORA_RANK


def test_ablations_do_not_repeat_the_final_models_own_point():
    """n=full at masked IS the final model. Running it again would spend an hour
    to produce a second copy of a row that already exists, and the data-size
    curve's right-hand end is that row."""
    curves = sweep.ablation_specs({"rank": 8, "learning_rate": 2e-4})
    assert all(s.get("data_size") is not None or s.get("masking") != "masked"
               for s in curves)
    assert not any(s.get("data_size") is None and s.get("masking") == "masked"
                   for s in curves)


def test_ablation_names_cannot_collide_with_the_final_model():
    """Adapters are keyed on the run name. An ablation named lora_r8 would
    overwrite the adapter every reported number is generated from."""
    winner = {"rank": 8, "learning_rate": 2e-4}
    names = [s["run"] for s in sweep.ablation_specs(winner)]
    final = {s["run"] for s in sweep.final_specs(winner)}
    assert len(set(names)) == len(names)
    assert not (set(names) & final)
    assert all(n.startswith("ablation_") for n in names)


def test_the_data_size_sweep_straddles_limas_threshold():
    """LIMA reported 1,000 curated examples sufficing. The comparison the
    project claims is where OUR curve flattens relative to theirs, which needs
    points on both sides of 1,000 — not merely that we also used few examples."""
    sizes = [s["data_size"] for s in sweep.ablation_specs(
        {"rank": 8, "learning_rate": 2e-4}) if s.get("data_size")]
    assert any(n < 1000 for n in sizes) and 1000 in sizes


def test_every_run_early_stops_via_the_global_default():
    """Early stopping is on globally, so no spec sets it. The 200-poem point
    would otherwise train to its step floor regardless of its validation curve,
    and the data-size curve would measure overfitting rather than data."""
    assert config.EARLY_STOPPING
    curves = sweep.ablation_specs({"rank": 8, "learning_rate": 2e-4})
    assert curves and not any("early_stopping" in s for s in curves)


# --- selection ----------------------------------------------------------------

def test_lowest_validation_loss_wins():
    best = sweep.select_winner([record("a", 2.0), record("b", 1.5, rank=4),
                                record("c", 1.8)])
    assert best["rank"] == 4


def test_ties_break_toward_fewer_parameters():
    """The direction Hu et al.'s saturation finding predicts, so the tiebreak
    cannot quietly favour the outcome."""
    best = sweep.select_winner([record("big", 1.5, params=4_000_000, rank=16),
                                record("small", 1.5, params=737_280, rank=8)])
    assert best["rank"] == 8


def test_nan_losses_are_excluded():
    """A diverged run records NaN, which would otherwise sort as a winner."""
    best = sweep.select_winner([record("diverged", float("nan"), rank=16),
                                record("ok", 2.0, rank=8)])
    assert best["rank"] == 8


def test_selection_needs_something_to_select_on():
    try:
        sweep.select_winner([{"run": "x", "final_val_loss": None,
                              "trainable_params": 1, "rank": 8,
                              "learning_rate": 2e-4}])
    except AssertionError:
        return
    raise AssertionError("selection proceeded with no usable metric")


def test_selection_metric_is_not_a_judge_score():
    """Judge scores are the OUTCOME. Selecting on them would fit the
    hyperparameters to the thing being reported."""
    assert "judge" not in config.SWEEP_SELECTION_METRIC
    assert "score" not in config.SWEEP_SELECTION_METRIC
    assert "gap" not in config.SWEEP_SELECTION_METRIC


# --- naming and planning ------------------------------------------------------

def test_run_names_encode_what_varied():
    assert sweep.run_name({"rank": 4, "learning_rate": 1e-4}) == "sweep_r4_lr0.0001"
    assert "n200" in sweep.run_name({"rank": 8, "learning_rate": 2e-4,
                                     "data_size": 200})
    assert "unmasked" in sweep.run_name({"rank": 8, "learning_rate": 2e-4,
                                         "masking": "unmasked"})


def test_plan_reports_the_full_size_before_any_gpu_time():
    plan = sweep.plan()
    assert plan["total"] == (plan["tuning_runs"] + plan["final_runs"]
                             + plan["ablation_runs"])
    configs = len(config.RANK_SWEEP) + len(config.LR_SWEEP) - 1
    assert plan["tuning_runs"] == configs * config.TUNING_FOLDS
    assert plan["final_runs"] == len(config.LORA_ARM_RANKS)


def test_specs_do_not_override_the_global_early_stopping_default():
    """An absent key must fall through to config, not silently disable early
    stopping. Passing False explicitly would turn it off for all nine grid runs
    while the config said it was on."""
    import inspect
    from src.train import sweep as module

    for runner in (module.run_cv_fold, module.run_final, module.run_ablation):
        source = inspect.getsource(runner)
        assert 'spec.get("early_stopping")' in source
        assert 'spec.get("early_stopping", False)' not in source


def test_curve_specs_leave_early_stopping_to_config():
    for spec in sweep.ablation_specs({"rank": 8, "learning_rate": 2e-4}):
        assert "early_stopping" not in spec


# --- smoke rows must never win ------------------------------------------------

def real_run(name, loss, model=None, **extra):
    return {"run": name, "model": model or config.MODEL, "rank": 8,
            "learning_rate": 2e-4, "trainable_params": 700_000,
            config.SWEEP_SELECTION_METRIC: loss, **extra}


def test_a_smoke_row_cannot_be_selected():
    """A five-step gpt2 run on six examples can post a LOWER validation loss
    than a real run. Selection is pre-registered on that metric alone, so
    nothing else in the function would notice it winning — and the returned
    'winner' would be a configuration that was never actually trained."""
    from src.train import sweep

    winner = sweep.select_winner([
        real_run("smoke_check", 0.01, model="gpt2", smoke=True),
        real_run("sweep_r8_lr0.0002", 1.80),
    ])
    assert winner["rank"] == 8 and winner["learning_rate"] == 2e-4


def test_a_row_from_another_model_cannot_be_selected():
    from src.train import sweep

    winner = sweep.select_winner([
        real_run("other", 0.01, model="some/other-model"),
        real_run("sweep_r4_lr0.0001", 1.9, rank=4, learning_rate=1e-4),
    ])
    assert winner["rank"] == 4


def test_selection_raises_when_every_row_is_ineligible():
    """Better than silently selecting the only smoke row present."""
    from src.train import sweep

    try:
        sweep.select_winner([real_run("smoke", 0.01, model="gpt2", smoke=True)])
    except AssertionError as error:
        assert "eligible" in str(error)
        return
    raise AssertionError("a smoke-only record set produced a winner")


# --- smoke runs cannot contaminate real results -------------------------------

def test_smoke_writes_to_separate_result_files():
    """Each of these has a route by which a smoke row does damage silently:
    runs.csv feeds selection, arm_outputs keys on (poem_id, arm) and takes the
    newest, and probe_all skips ids already present."""
    import os
    import subprocess
    import sys

    script = ("import config; print(config.RUNS_CSV_PATH.name, "
              "config.ARM_OUTPUTS_PATH.name, config.CONTAMINATION_PATH.name)")
    names = {}
    for smoke in ("0", "1"):
        result = subprocess.run([sys.executable, "-c", script],
                                env={**os.environ, "SMOKE": smoke},
                                capture_output=True, text=True, check=True)
        names[smoke] = result.stdout.split()

    assert not set(names["0"]) & set(names["1"])
    assert all(n.startswith("smoke_") for n in names["1"])


def test_the_gpu_only_modules_are_smoke_runnable():
    """`SMOKE=1 python -m src.train.loop` and `... src.generate.inference` are
    required by the project rules: these two never otherwise run locally, so a
    bug in either costs a Kaggle session to discover."""
    import inspect

    from src.generate import inference
    from src.train import loop

    for module in (loop, inference):
        source = inspect.getsource(module)
        assert '__name__ == "__main__"' in source, f"{module.__name__} has no entrypoint"
        assert "_smoke" in source


# --- the driver ---------------------------------------------------------------

def test_there_is_exactly_one_adapter_per_arm():
    """The whole point of replacing 5-fold with a holdout. Five adapters trained
    on different data cannot be merged, so the README's load snippet could not
    name one — and an arm the results report has to be a single artefact."""
    specs = sweep.final_specs({"rank": 4, "learning_rate": 5e-4})
    assert [s["run"] for s in specs] == ["lora_r8", "lora_r16"]
    assert not any("fold" in s for s in specs)


def test_the_arm_ranks_ignore_the_winning_rank():
    """They name the pre-registered arms. Letting a winning rank of 4 rename
    them would change the experiment after seeing results — the rank question is
    answered by the CV curve over {4, 8, 16}, not by renaming an arm."""
    specs = sweep.final_specs({"rank": 4, "learning_rate": 1e-4})
    assert [s["rank"] for s in specs] == list(config.LORA_ARM_RANKS)
    assert all(isinstance(s["rank"], int) for s in specs)
    # The learning rate IS taken from the winner.
    assert {s["learning_rate"] for s in specs} == {1e-4}


def test_both_arms_see_the_same_data():
    """lora_r8 vs lora_r16 must isolate adapter capacity. Under 5-fold it
    compared fold-0 r8 against fold-0 r16 and the difference also carried which
    poems each had seen; both now train on the whole pool."""
    import inspect

    specs = sweep.final_specs({"learning_rate": 2e-4})
    assert len({s["learning_rate"] for s in specs}) == 1
    # run_final takes the pool unconditionally — no per-spec data selection.
    source = inspect.getsource(sweep.run_final)
    assert "pool_partition(pairs)" in source
    assert "data_size" not in source


def test_completed_runs_are_skipped(tmp_path, monkeypatch):
    """19 runs will not reliably fit one Kaggle session. A session that dies at
    run 14 must resume at 15, not restart at 1."""
    import csv

    from src.train import sweep

    path = tmp_path / "runs.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "final_val_loss"])
        writer.writeheader()
        writer.writerow({"run": "sweep_r8_lr0.0002", "final_val_loss": 1.6})
    monkeypatch.setattr(config, "RUNS_CSV_PATH", path)

    assert set(sweep.load_completed()) == {"sweep_r8_lr0.0002"}


def test_completed_is_empty_when_nothing_has_run(tmp_path, monkeypatch):
    from src.train import sweep

    monkeypatch.setattr(config, "RUNS_CSV_PATH", tmp_path / "absent.csv")
    assert sweep.load_completed() == {}


def test_every_run_keeps_its_weights():
    """On by default. Each row in runs.csv is then traceable to the model that
    produced it, and a sweep configuration that later matters can be inspected
    rather than retrained."""
    import inspect

    from src.train import sweep

    for runner in (sweep.run_cv_fold, sweep.run_final, sweep.run_ablation):
        assert "save_adapter: bool = True" in inspect.getsource(runner)


def test_adapter_paths_never_collide_across_the_programme():
    """The reason weights were not saved before. Three learning rates at rank 8
    all resolve to the same (rank, fold), so keying on that would leave one file
    per rank belonging to no recorded run in particular. Keying on the run name
    — which encodes everything that varied — fixes it."""
    from src.train import sweep

    winner = {"rank": 16, "learning_rate": 5e-4}
    names = ([s["run"] for s in sweep.cv_specs()]
             + [s["run"] for s in sweep.ablation_specs(winner)]
             + [s["run"] for s in sweep.final_specs(winner)])
    paths = [config.run_adapter_dir(n) for n in names]
    assert len(set(paths)) == len(set(names)), "two runs share an adapter path"


def test_the_final_adapter_lands_where_its_run_name_says():
    """One model, one adapter directory, named after the run — which is what
    the README's load snippet points at."""
    from src.train import sweep

    for spec in sweep.final_specs({"rank": 8, "learning_rate": 2e-4}):
        path = config.run_adapter_dir(spec["run"])
        # `smoke_` under SMOKE, and deliberately so — a five-step gpt2 adapter
        # must not land on the path the README tells a reader to load.
        assert path.name.endswith(f"lora_r{spec['rank']}")
        assert config.SMOKE == path.name.startswith("smoke_")


def test_csv_numbers_are_coerced_before_selection():
    """runs.csv round-trips everything as strings, and '10.0' < '9.0' is True
    under string ordering — which would silently pick the wrong winner."""
    from src.train import sweep

    rows = sweep.coerce([{"run": "a", "final_val_loss": "1.62",
                          "trainable_params": "4399104", "rank": "8",
                          "learning_rate": "0.0002"}])
    assert rows[0]["final_val_loss"] == 1.62
    assert rows[0]["trainable_params"] == 4399104.0


def test_the_full_programme_fits_a_gpu_week():
    """Derived from config, not hardcoded: the sweep has been narrowed once for
    compute already, and a test that pins a count would break on the next such
    decision rather than checking the property that matters.

    At ~1.6 GPU-hours per full-corpus run, a 30-hour week bounds the programme
    at roughly 18 runs. The 19-run version did not finish."""
    from src.train import sweep

    plan = sweep.plan()
    configs = len(config.RANK_SWEEP) + len(config.LR_SWEEP) - 1
    expected = (configs * config.TUNING_FOLDS + len(config.LORA_ARM_RANKS)
                + plan["ablation_runs"])
    assert plan["total"] == expected, f"programme is {plan['total']}"

    # Tuning runs on a subsample of the pool and ablations on subsets, so the
    # per-run hour is only paid in full by the final model. Costing everything
    # at the full-corpus rate is therefore an upper bound, which is the side to
    # be wrong on when the question is whether the week is enough.
    assert plan["total"] * 1.6 < 30, f"{plan['total']} runs is over a GPU week"


def test_a_name_match_at_the_wrong_config_is_not_done():
    """The day-3 pilot is recorded as lora_r8_fold0 at the DEFAULT learning
    rate, which is exactly the name stage 3 gives its fold-0 run. If the sweep
    picks a different rate, skipping on the name alone would keep the pilot's
    adapter and ship a final result trained at a configuration the sweep
    rejected — with the row present, the adapter present, and the name right.
    """
    from src.train import sweep

    pilot = {"run": "lora_r8_fold0", "rank": "8", "learning_rate": "0.0002"}
    completed = {"lora_r8_fold0": pilot}

    same = {"run": "lora_r8_fold0", "rank": 8, "learning_rate": 2e-4, "fold": 0}
    other = {"run": "lora_r8_fold0", "rank": 8, "learning_rate": 5e-4, "fold": 0}

    assert sweep.already_done(same, completed) is True
    assert sweep.already_done(other, completed) is False


def test_a_run_never_recorded_is_not_done():
    from src.train import sweep

    assert sweep.already_done({"rank": 8, "learning_rate": 2e-4}, {}) is False


def test_a_row_missing_its_hyperparameters_is_re_run():
    """Better to repeat a run than to trust a row that cannot prove what it
    was."""
    from src.train import sweep

    completed = {"sweep_r8_lr0.0002": {"run": "sweep_r8_lr0.0002",
                                       "rank": "", "learning_rate": ""}}
    assert sweep.already_done({"rank": 8, "learning_rate": 2e-4},
                              completed) is False


def test_a_row_missing_a_required_column_is_re_run():
    """The day-3 pilot is recorded as lora_r8_fold0 at the default learning
    rate but predates heldout_perplexity. If the sweep picks that same config,
    matching on name and hyperparameters would skip the final run and leave H4
    without a value for that fold — silently."""
    from src.train import sweep

    pilot = {"run": "lora_r8_fold0", "rank": "8", "learning_rate": "0.0002",
             "heldout_perplexity": ""}
    spec = {"run": "lora_r8_fold0", "rank": 8, "learning_rate": 2e-4, "fold": 0}

    assert sweep.already_done(spec, {"lora_r8_fold0": pilot}) is True


def test_selection_refuses_to_read_the_held_out_fold():
    """The held-out fold is measured once per run and recorded, which is safe
    ONLY because nothing is chosen on it. Select on it and it stops being a test
    set — every number later reported from it would carry exactly the selection
    bias that measuring it separately exists to avoid.
    """
    from src.train import sweep

    original = config.SWEEP_SELECTION_METRIC
    try:
        config.SWEEP_SELECTION_METRIC = "heldout_perplexity"
        sweep.select_winner([real_run("a", 1.0)])
    except AssertionError as error:
        assert "turns it into a validation set" in str(error)
        return
    finally:
        config.SWEEP_SELECTION_METRIC = original
    raise AssertionError("selection on the held-out fold was permitted")


def test_the_pre_registered_metric_is_a_validation_one():
    assert "heldout" not in config.SWEEP_SELECTION_METRIC
    assert config.SWEEP_SELECTION_METRIC == "final_val_loss"


def test_the_winner_survives_a_csv_round_trip():
    """The bug this catches: coerce() cast every numeric column with float(),
    so rank 16 became 16.0, propagated into LoraConfig(r=16.0), and died inside
    peft at nn.Linear(in_features, 16.0) — several frames deep, naming neither
    the rank nor the sweep. It also renamed the run `sweep_r16.0_...`, so the
    stage would not have matched its own rows on a resume.

    Earlier tests missed it by calling select_winner with dict literals holding
    int ranks, never with rows that had been through coerce.
    """
    from src.model import setup
    from src.train import sweep

    as_csv = [{"run": "sweep_r16_lr0.0002", "model": config.MODEL,
               "rank": "16", "learning_rate": "0.0002",
               "trainable_params": "8798208", "final_val_loss": "1.55"}]
    winner = sweep.select_winner(sweep.coerce(as_csv))

    assert isinstance(winner["rank"], int)
    assert isinstance(winner["learning_rate"], float)
    # The name must not carry a decimal point, or resume matching breaks.
    assert "." not in sweep.run_name(winner).split("_lr")[0]
    # And it must be usable where it is actually consumed.
    setup.lora_config(winner["rank"])


def test_a_float_rank_is_refused_at_the_point_of_use():
    from src.model import setup

    try:
        setup.lora_config(16.0)
    except AssertionError as error:
        assert "must be an int" in str(error)
        return
    raise AssertionError("a float rank reached LoraConfig")


def test_curve_specs_inherit_an_integer_rank():
    from src.train import sweep

    as_csv = [{"run": "sweep_r16_lr0.0002", "model": config.MODEL, "rank": "16",
               "learning_rate": "0.0002", "trainable_params": "8798208",
               "final_val_loss": "1.55"}]
    winner = sweep.select_winner(sweep.coerce(as_csv))
    for spec in sweep.ablation_specs(winner) + sweep.final_specs(winner):
        assert isinstance(spec["rank"], int), spec


def test_tuning_varies_one_axis_at_a_time():
    """CLAUDE.md specifies one-at-a-time and says explicitly not to run the full
    grid. A previous version ran the 3x3 product: 11.6 GPU-hours for tuning
    alone, which does not fit a 12-hour Kaggle session. One session was lost."""
    from src.train import sweep

    specs = sweep.tuning_grid()
    assert len(specs) == len(config.RANK_SWEEP) + len(config.LR_SWEEP) - 1

    # Every spec differs from the default in at most one axis.
    for spec in specs:
        varied = sum([spec["rank"] != config.LORA_RANK,
                      spec["learning_rate"] != config.LEARNING_RATE])
        assert varied <= 1, spec


def test_the_shared_default_is_trained_once():
    from src.train import sweep

    names = [sweep.run_name(s) for s in sweep.tuning_grid()]
    assert len(names) == len(set(names))


def test_both_sweep_axes_are_still_covered():
    """Cheaper must not mean incomplete: the rank curve is compared against Hu
    et al.'s saturation finding and cannot lose a point."""
    from src.train import sweep

    specs = sweep.tuning_grid()
    assert {s["rank"] for s in specs} == set(config.RANK_SWEEP)
    assert {s["learning_rate"] for s in specs} == set(config.LR_SWEEP)


def test_batching_bounds_what_a_lost_session_costs():
    """RESULTS_DIR does not survive a session on Kaggle or Colab. A stage that
    runs for hours and is cut off loses every row it wrote."""
    import inspect

    from src.train import sweep

    for stage in (sweep.run_cv_stage, sweep.run_ablation_stage):
        assert "max_runs" in inspect.signature(stage).parameters


# --- tuning is sequential, not independent ------------------------------------

def test_the_rank_sweep_runs_at_the_chosen_learning_rate():
    """Greedy coordinate descent, not independent one-at-a-time. Sweeping rank
    around a fixed DEFAULT would draw the rank curve at a configuration the
    search may already have rejected — the same objection that puts the
    ablations at the winner rather than at the defaults."""
    from src.train import sweep

    chosen = next(lr for lr in config.LR_SWEEP if lr != config.LEARNING_RATE)
    for spec in sweep.rank_specs(chosen):
        assert spec["learning_rate"] == chosen


def test_the_lr_sweep_covers_every_rate_at_one_rank():
    from src.train import sweep

    specs = sweep.lr_specs()
    assert {s["learning_rate"] for s in specs} == set(config.LR_SWEEP)
    assert {s["rank"] for s in specs} == {config.LORA_RANK}


def test_the_shared_point_is_trained_once_across_the_two_stages():
    """r8 at the winning rate appears in both stages; run_stage must skip the
    second occurrence or the sweep costs six runs instead of five."""
    from src.train import sweep

    # Whatever 1a picks comes FROM lr_specs, so the shared point always exists.
    chosen = config.LR_SWEEP[-1]
    done = {sweep.run_name(s): {"run": sweep.run_name(s), "rank": s["rank"],
                                "learning_rate": s["learning_rate"]}
            for s in sweep.lr_specs()}
    repeats = [s for s in sweep.rank_specs(chosen) if sweep.already_done(s, done)]
    assert len(repeats) == 1 and repeats[0]["rank"] == config.LORA_RANK


def test_the_sequential_design_costs_no_more_than_independent():
    from src.train import sweep

    assert len(sweep.tuning_grid()) == len(config.RANK_SWEEP) + len(config.LR_SWEEP) - 1


def test_selecting_from_an_incomplete_stage_raises(tmp_path, monkeypatch):
    """Stage 1b sweeps rank AT THE RATE 1a chose, so a partial 1a means the
    whole rank sweep may sit at a configuration the missing runs would have
    beaten. This happened: max_runs=2 stopped 1a at two of three rates, the
    notebook selected from those two, and 1b ran to completion at it."""
    import csv

    from src.train import sweep

    path = tmp_path / "runs.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "rank", "learning_rate"])
        writer.writeheader()
        for lr in list(config.LR_SWEEP)[:-1]:          # one rate short
            writer.writerow({"run": f"sweep_r{config.LORA_RANK}_lr{lr:g}",
                             "rank": config.LORA_RANK, "learning_rate": lr})
    monkeypatch.setattr(config, "RUNS_CSV_PATH", path)

    try:
        sweep.assert_stage_complete(sweep.lr_specs(), "stage 1a")
    except AssertionError as error:
        assert "incomplete" in str(error)
        return
    raise AssertionError("selection was allowed from a partial stage")


def test_a_complete_stage_passes(tmp_path, monkeypatch):
    import csv

    from src.train import sweep

    path = tmp_path / "runs.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "rank", "learning_rate"])
        writer.writeheader()
        for spec in sweep.lr_specs():
            writer.writerow({"run": sweep.run_name(spec), "rank": spec["rank"],
                             "learning_rate": spec["learning_rate"]})
    monkeypatch.setattr(config, "RUNS_CSV_PATH", path)

    sweep.assert_stage_complete(sweep.lr_specs(), "stage 1a")   # must not raise


def test_the_batch_limit_does_not_inflate_the_recorded_count(caplog, tmp_path,
                                                             monkeypatch):
    """Counting after the truncation treated the untouched remainder as
    finished: with 3 of 6 recorded and max_runs=1 the log announced "5 already
    recorded, 1 to run", which reads as almost done rather than half done."""
    import csv
    import logging

    from src.train import sweep

    path = tmp_path / "runs.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "rank",
                                                    "learning_rate"])
        writer.writeheader()
        winner = {"rank": 8, "learning_rate": 1e-4}
        for spec in sweep.ablation_specs(winner)[:3]:   # 3 of the 4 ablations
            writer.writerow({"run": spec["run"], "rank": 8,
                             "learning_rate": 1e-4})
    monkeypatch.setattr(config, "RUNS_CSV_PATH", path)
    monkeypatch.setattr(sweep, "run_ablation", lambda *a, **k: None)

    with caplog.at_level(logging.INFO):
        sweep.run_ablation_stage([], {"rank": 8, "learning_rate": 1e-4},
                                 max_runs=1)

    line = next(m for m in caplog.messages if "already recorded" in m)
    assert "3 already recorded" in line, line
    assert "1 still to run" in line, line


# --- ablations ----------------------------------------------------------------
#
# The data-size curve is a comparison across runs, so what has to hold is that
# only the size differs. Anything else that moves with it — which poems, which
# split, which set it is scored on — turns the curve into a measurement of two
# things at once.

def ablation_pairs(n=40):
    """A corpus with a test partition and a pool, shaped like the real one."""
    return [{"poem_id": i, "author": f"a{i % 8}", "poem": "line\n" * 9,
             "interpretation": "x"} for i in range(n)]


def stub_splits(monkeypatch, pairs, test_ids):
    """Route pool/test through explicit id sets, so the assertions in the
    runner are exercised without holdout.json being present."""
    from src.data import splits

    test = [p for p in pairs if p["poem_id"] in test_ids]
    pool = [p for p in pairs if p["poem_id"] not in test_ids]
    monkeypatch.setattr(splits, "test_partition", lambda _: test)
    monkeypatch.setattr(splits, "pool_partition", lambda _: pool)
    monkeypatch.setattr(splits, "untouched_authors", lambda _: set())
    return pool, test


def capture_train(monkeypatch):
    """Record what reaches loop.train instead of training anything."""
    from src.model import setup
    from src.train import dataset, loop

    seen = {}

    def fake_train(model, tokenizer, examples, pairs, **kwargs):
        seen.update(kwargs, examples=examples)
        return {"run": kwargs["run_name"]}

    monkeypatch.setattr(loop, "train", fake_train)
    monkeypatch.setattr(dataset, "build_dataset", lambda p, t: list(p))
    monkeypatch.setattr(setup, "load_tokenizer", lambda: None)
    monkeypatch.setattr(setup, "load_base_model", lambda: None)
    monkeypatch.setattr(setup, "apply_lora", lambda m, rank: m)
    monkeypatch.setattr(setup, "assert_matched_learning_rate",
                        lambda *a, **k: None)
    return seen


def test_the_data_size_subset_is_deterministic(monkeypatch):
    """Two runs at n=10 must train on the SAME ten poems. Subsetting whatever
    order the corpus arrived in would make the curve depend on file order, and
    a re-run at one point would silently not be comparable with the others."""
    from src.train import sweep

    pairs = ablation_pairs()
    first, second = [], []
    for out in (first, second):
        stub_splits(monkeypatch, pairs, test_ids={0, 1, 2})
        seen = capture_train(monkeypatch)
        sweep.run_ablation({"rank": 8, "learning_rate": 2e-4, "data_size": 10,
                            "run": "ablation_n10"}, pairs, save_adapter=False)
        out.extend(p["poem_id"] for p in seen["examples"])

    assert first == second == sorted(first)
    assert len(first) == 10


def test_the_data_size_subset_never_reaches_into_the_test_set(monkeypatch):
    """The subset is taken from the pool, so a low-numbered test poem must not
    be swept up by `sorted(...)[:n]`. It would train on data the same run is
    then scored against, and the smallest data point — the one most likely to
    look surprisingly good — is where it would happen."""
    from src.train import sweep

    pairs = ablation_pairs()
    pool, test = stub_splits(monkeypatch, pairs, test_ids={0, 1, 2, 3, 4})
    seen = capture_train(monkeypatch)
    sweep.run_ablation({"rank": 8, "learning_rate": 2e-4, "data_size": 6,
                        "run": "ablation_n6"}, pairs, save_adapter=False)

    trained = {p["poem_id"] for p in seen["examples"]}
    assert not trained & {p["poem_id"] for p in test}
    assert trained == {5, 6, 7, 8, 9, 10}


def test_every_ablation_is_scored_on_the_same_test_set(monkeypatch):
    """The data-size curve's right-hand end is the final model's own row. If an
    ablation were scored on anything else, the curve would join points measured
    by different rulers."""
    from src.train import sweep

    pairs = ablation_pairs()
    winner = {"rank": 8, "learning_rate": 2e-4}
    for spec in sweep.ablation_specs(winner):
        _, test = stub_splits(monkeypatch, pairs, test_ids={0, 1, 2})
        seen = capture_train(monkeypatch)
        sweep.run_ablation({**spec, "data_size": spec.get("data_size") and 5},
                           pairs, save_adapter=False)
        assert [p["poem_id"] for p in seen["heldout"]] == \
            [p["poem_id"] for p in test], spec["run"]
        assert seen.get("fold") is None, "an ablation is not a fold"


def test_the_unmasked_ablation_trains_on_the_whole_pool(monkeypatch):
    """It varies masking and nothing else, so it must see exactly what the
    final model sees."""
    from src.train import sweep

    pairs = ablation_pairs()
    pool, _ = stub_splits(monkeypatch, pairs, test_ids={0, 1, 2})
    seen = capture_train(monkeypatch)
    sweep.run_ablation({"rank": 8, "learning_rate": 2e-4,
                        "masking": "unmasked", "run": "ablation_unmasked"},
                       pairs, save_adapter=False)

    assert len(seen["examples"]) == len(pool)
    assert seen["masking"] == "unmasked"


def test_an_ablation_refuses_to_train_on_the_test_set(monkeypatch):
    """The guard that would have to fail silently for the whole result to be
    wrong. If pool and test ever overlap, the run is scored by data it trained
    on and every ablation number is optimistic."""
    from src.data import splits
    from src.train import sweep

    pairs = ablation_pairs()
    capture_train(monkeypatch)
    monkeypatch.setattr(splits, "untouched_authors", lambda _: set())
    monkeypatch.setattr(splits, "pool_partition", lambda _: pairs)
    monkeypatch.setattr(splits, "test_partition", lambda _: pairs[:3])

    try:
        sweep.run_ablation({"rank": 8, "learning_rate": 2e-4, "data_size": 5,
                            "run": "ablation_n5"}, pairs, save_adapter=False)
    except AssertionError as error:
        assert "test set" in str(error)
        return
    raise AssertionError("an ablation trained on the poems scoring it")
