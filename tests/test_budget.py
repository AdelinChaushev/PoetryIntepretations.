"""Tests for the epoch-based budget and early stopping.

Two failures matter. A budget that does not scale with dataset size makes the
data-size curve measure training length rather than data. And early stopping
without restoring the best weights SAVES the overfitted model it stopped
because of — the run ends `patience` evaluations past its own best.
"""

from __future__ import annotations

import config


def unclamped(n: int) -> bool:
    """Whether `n` examples land strictly between the floor and the ceiling.

    SMOKE sets both to a handful of steps, so every dataset clamps and the
    scaling behaviour is deliberately flattened. Tests that assert scaling say
    so rather than asserting a property the configuration has switched off.
    """
    return config.MIN_TRAIN_STEPS < config.steps_for(n) < config.MAX_STEPS


def test_budget_scales_with_dataset_size():
    """Equal EPOCHS, not equal steps: a run on 500 poems should take fewer
    updates than one on 1,786 — where neither is clamped."""
    if not (unclamped(500) and unclamped(1786)):
        return  # SMOKE clamps both; nothing to observe
    assert config.steps_for(500) < config.steps_for(1786)


def test_epochs_are_roughly_constant_above_the_floor():
    sizes = [n for n in (500, 1000, 1786) if unclamped(n)]
    if len(sizes) < 2:
        return
    eff = config.BATCH_SIZE * config.GRAD_ACCUM_STEPS
    epochs = [config.steps_for(n) * eff / n for n in sizes]
    assert max(epochs) - min(epochs) < 0.5


def test_small_datasets_hit_the_floor_not_a_useless_budget():
    """Nine epochs of 200 poems is ~113 steps, of which 10% is warmup. A run
    that short is undertrained rather than data-limited, so the floor applies."""
    eff = config.BATCH_SIZE * config.GRAD_ACCUM_STEPS
    raw = config.MAX_EPOCHS * (200 // eff)
    if raw >= config.MIN_TRAIN_STEPS:
        return  # this configuration does not need the floor at 200
    assert config.steps_for(200) == config.MIN_TRAIN_STEPS


def test_the_budget_always_respects_both_bounds():
    """True in every mode, which is what makes it worth asserting."""
    for n in (1, 200, 1786, 10**6):
        assert config.MIN_TRAIN_STEPS <= config.steps_for(n) <= config.MAX_STEPS


def test_large_datasets_hit_the_ceiling():
    assert config.steps_for(10**6) == config.MAX_STEPS


def test_floor_and_ceiling_do_not_cross():
    assert config.MIN_TRAIN_STEPS < config.MAX_STEPS


def test_early_stopping_is_on_by_default():
    """Each configuration is compared at ITS best, not at an arbitrary shared
    step count."""
    assert config.EARLY_STOPPING


def test_best_weights_are_restored():
    assert config.RESTORE_BEST_WEIGHTS


def test_run_record_reports_what_the_run_actually_did():
    """A budget that early-stops means the configured steps are not the steps
    run. Both belong in runs.csv or the difference is invisible."""
    import inspect
    from src.train import loop

    source = inspect.getsource(loop.train)
    for field in ('"epochs_run"', '"steps_run"', '"max_steps"',
                  '"early_stopped"', '"early_stopping_enabled"',
                  '"best_checkpoint"', '"val_perplexity"'):
        assert field in source, f"runs.csv is missing {field}"


# --- the library does the loop ------------------------------------------------

def test_training_uses_the_library_trainer():
    """Gradient accumulation, scheduling, clipping, checkpointing, early
    stopping, best-weight restoration AND label masking are all standard
    components with standard bugs already found in them."""
    import inspect
    from src.train import loop

    source = inspect.getsource(loop)
    assert "SFTTrainer" in source
    assert "EarlyStoppingCallback" in source
    assert "load_best_model_at_end" in source


def test_best_model_is_restored_by_the_trainer():
    """Without it a run ends `patience` evaluations past its own best and would
    save the overfitted model it stopped because of."""
    import inspect
    from src.train import loop

    source = inspect.getsource(loop.training_arguments)
    assert "load_best_model_at_end=config.RESTORE_BEST_WEIGHTS" in source
    assert 'metric_for_best_model="eval_loss"' in source
    assert "greater_is_better=False" in source


def test_masking_is_delegated_to_the_library():
    """SFTTrainer masks the prompt for a prompt-completion dataset. Two earlier
    versions of this module hand-rolled first the loop and then the labels; the
    loop version shipped a bug that silently disabled early stopping across a
    whole sweep."""
    import inspect
    from src.train import loop

    source = inspect.getsource(loop.training_arguments)
    assert "completion_only_loss=True" in source
    assert "from trl import SFTConfig" in inspect.getsource(loop.training_arguments)


def test_warmup_ratio_is_converted_not_hardcoded():
    """This transformers version has no warmup_ratio. A hardcoded step count
    would silently change meaning whenever the budget moved."""
    import inspect
    from src.train import loop

    source = inspect.getsource(loop.training_arguments)
    assert "config.WARMUP_RATIO" in source


def test_no_custom_collator_remains():
    """TRL builds the batch and the labels. A leftover custom collator would
    quietly take that job back."""
    from src.train import loop

    assert not hasattr(loop, "Collator")


def test_the_length_filter_stayed_ours():
    """TRL truncates to max_length; the project drops instead, because a
    truncated poem is still scored against its full text by the grounding
    checker."""
    import inspect
    from src.train import dataset

    assert "DROPPED, not" in inspect.getsource(dataset.build_dataset)


def test_effective_batch_is_unchanged_by_the_split():
    """4x4 and 8x2 give the same update; only peak memory differs."""
    assert config.BATCH_SIZE * config.GRAD_ACCUM_STEPS == 16 or config.SMOKE
