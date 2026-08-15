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
