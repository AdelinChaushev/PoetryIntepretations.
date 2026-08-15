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

def test_tuning_is_a_product_not_one_at_a_time():
    """Rank and learning rate interact — alpha/r scaling and lr both control how
    far the adapter moves per step — so the best rank at one lr need not be the
    best at another."""
    grid = sweep.tuning_grid()
    assert len(grid) == len(config.RANK_SWEEP) * len(config.LR_SWEEP)
    assert len({(s["rank"], s["learning_rate"]) for s in grid}) == len(grid)


def test_curves_run_at_the_winner_not_the_defaults():
    """A curve drawn at a rejected configuration describes a model nobody will
    train."""
    winner = {"rank": 16, "learning_rate": 5e-4}
    for spec in sweep.reported_curves(winner):
        assert spec["rank"] == 16 and spec["learning_rate"] == 5e-4


def test_curves_do_not_repeat_the_winners_own_point():
    curves = sweep.reported_curves({"rank": 8, "learning_rate": 2e-4})
    assert all(s.get("data_size") is not None or s.get("masking") != "masked"
               for s in curves)
    assert not any(s.get("data_size") is None and s.get("masking") == "masked"
                   for s in curves)


def test_every_run_early_stops_via_the_global_default():
    """Early stopping is on globally, so no spec sets it. The 200-poem point
    would otherwise train to its step floor regardless of its validation curve,
    and the data-size curve would measure overfitting rather than data."""
    assert config.EARLY_STOPPING
    curves = sweep.reported_curves({"rank": 8, "learning_rate": 2e-4})
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
    assert plan["total"] == plan["tuning_runs"] + plan["curve_runs"]
    assert plan["tuning_runs"] == len(config.RANK_SWEEP) * len(config.LR_SWEEP)


def test_specs_do_not_override_the_global_early_stopping_default():
    """An absent key must fall through to config, not silently disable early
    stopping. Passing False explicitly would turn it off for all nine grid runs
    while the config said it was on."""
    import inspect
    from src.train import sweep as module

    source = inspect.getsource(module.run_one)
    assert 'spec.get("early_stopping")' in source
    assert 'spec.get("early_stopping", False)' not in source


def test_curve_specs_leave_early_stopping_to_config():
    for spec in sweep.reported_curves({"rank": 8, "learning_rate": 2e-4}):
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

def test_final_specs_are_the_six_reported_runs():
    from src.train import sweep

    specs = sweep.final_specs({"rank": 4, "learning_rate": 5e-4})
    assert len(specs) == config.N_FOLDS + 1
    assert sorted(s["fold"] for s in specs if s["rank"] == 8) == \
        list(range(config.N_FOLDS))
    r16 = [s for s in specs if s["rank"] == 16]
    assert len(r16) == 1 and r16[0]["fold"] == config.SINGLE_SPLIT_FOLD


def test_final_ranks_ignore_the_winning_rank():
    """The arms are named lora_r8 and lora_r16 in the pre-registration. Letting
    a winning rank of 4 rename them would change the experiment after seeing
    results — the rank question is answered by the reported curve instead."""
    from src.train import sweep

    specs = sweep.final_specs({"rank": 4, "learning_rate": 1e-4})
    assert {s["rank"] for s in specs} == {8, 16}
    # The learning rate IS taken from the winner.
    assert {s["learning_rate"] for s in specs} == {1e-4}


def test_r16_shares_a_fold_with_an_r8_run():
    """Otherwise lora_r8 vs lora_r16 compares a fold-averaged r8 against a
    single-split r16, and the difference includes which poems each saw."""
    from src.train import sweep

    specs = sweep.final_specs({"learning_rate": 2e-4})
    r16 = next(s for s in specs if s["rank"] == 16)
    assert any(s["rank"] == 8 and s["fold"] == r16["fold"] for s in specs)


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


def test_sweep_runs_do_not_save_adapters():
    """Nine grid points at three ranks share three adapter paths, so saving
    would leave one file per rank belonging to no recorded run in particular.
    Only the final runs' weights are ever loaded."""
    import inspect

    from src.train import sweep

    assert "save_adapters: bool = False" in inspect.getsource(sweep.run_stage)


def test_csv_numbers_are_coerced_before_selection():
    """runs.csv round-trips everything as strings, and '10.0' < '9.0' is True
    under string ordering — which would silently pick the wrong winner."""
    from src.train import sweep

    rows = sweep.coerce([{"run": "a", "final_val_loss": "1.62",
                          "trainable_params": "4399104", "rank": "8",
                          "learning_rate": "0.0002"}])
    assert rows[0]["final_val_loss"] == 1.62
    assert rows[0]["trainable_params"] == 4399104.0


def test_the_full_programme_is_19_runs():
    from src.train import sweep

    plan = sweep.plan()
    assert plan["total"] == 13
    assert plan["total"] + len(sweep.final_specs({})) == 19


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
    import inspect
    assert "requires" in inspect.signature(sweep.run_stage).parameters
