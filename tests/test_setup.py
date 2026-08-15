"""Tests for model loading and adapter attachment.

The assertion that matters is `assert_adapters_only`. Both failures it catches
are silent: a target_modules that matches nothing produces a completed run with
a flat loss curve and an empty adapter — which looks exactly like "fine-tuning
did not help", the project's own H2 prediction — and a base weight left
trainable makes the run a partial full fine-tune reported as LoRA.
"""

from __future__ import annotations

import config
from src.model import setup


class FakeParam:
    def __init__(self, n, requires_grad):
        self._n, self.requires_grad = n, requires_grad

    def numel(self):
        return self._n


class FakeModel:
    def __init__(self, named):
        self._named = named

    def named_parameters(self):
        return iter(self._named)

    def parameters(self):
        return (p for _, p in self._named)


def model(adapter_params=1000, frozen_params=1_000_000, leaked=0):
    named = [("base.layer.weight", FakeParam(frozen_params, False)),
             ("base.layer.lora_A.weight", FakeParam(adapter_params, True))]
    if leaked:
        named.append(("base.layer.mlp.weight", FakeParam(leaked, True)))
    if adapter_params == 0:
        named = [n for n in named if "lora_" not in n[0]]
    return FakeModel(named)


# --- the guard ---------------------------------------------------------------

def test_adapters_only_passes_on_a_correct_model():
    setup.assert_adapters_only(model())


def test_zero_trainable_parameters_raises():
    """target_modules matching nothing is the dangerous case: the run completes,
    the curve is flat, and the result reads as 'LoRA did not help'."""
    try:
        setup.assert_adapters_only(model(adapter_params=0))
    except AssertionError as error:
        assert "matched nothing" in str(error)
        return
    raise AssertionError("a model with no trainable parameters was accepted")


def test_a_trainable_base_weight_raises():
    """Otherwise the run is a partial full fine-tune reported as LoRA."""
    try:
        setup.assert_adapters_only(model(leaked=500))
    except AssertionError as error:
        assert "full fine-tune" in str(error)
        return
    raise AssertionError("a trainable base weight was accepted")


def test_parameter_counts():
    counts = setup.trainable_parameters(model(adapter_params=1000,
                                              frozen_params=999_000))
    assert counts["trainable"] == 1000
    assert counts["total"] == 1_000_000
    assert abs(counts["percent"] - 0.1) < 1e-9


# --- the rank/alpha relationship ---------------------------------------------

def test_alpha_scales_with_rank():
    """Fixed alpha would vary the alpha/r scaling 4x, 2x, 1x across the rank
    sweep, so the axis would measure capacity AND step size together."""
    scalings = {config.lora_alpha(r) / r for r in config.RANK_SWEEP}
    assert len(scalings) == 1


def test_alpha_matches_the_configured_multiplier():
    assert config.lora_alpha(8) == 8 * config.LORA_ALPHA_MULTIPLIER


# --- environment guards -------------------------------------------------------

def test_device_never_assumes_cuda():
    assert setup.device() in {"cpu", "cuda", "mps"}


def test_smoke_targets_match_the_smoke_model():
    """gpt2 has a fused c_attn instead of separate projections; the Qwen names
    would match nothing and train zero parameters."""
    if config.SMOKE:
        assert config.LORA_TARGET_MODULES == ("c_attn",)
    else:
        assert "q_proj" in config.LORA_TARGET_MODULES


# --- precision must agree between the model and the trainer --------------------

def test_bf16_support_is_the_strict_ampere_check():
    """torch.cuda.is_bf16_supported() counts EMULATION and returns True on
    Turing; TrainingArguments requires Ampere and raises otherwise.

    Observed on a Kaggle T4: the model loaded as bfloat16 and SFTConfig then
    refused to construct. Using the loose check for the dtype and letting
    transformers apply the strict one to the trainer is the whole bug.
    """
    import inspect

    from src.model import setup

    source = inspect.getsource(setup.supports_bf16)
    assert "get_device_capability" in source
    assert "major >= 8" in source


def test_the_model_dtype_and_the_trainer_agree():
    """One helper decides both, so a card can never get bf16 weights and an
    fp16 trainer — or the reverse."""
    import inspect

    from src.train import loop

    source = inspect.getsource(loop.training_arguments)
    assert "setup.supports_bf16()" in source
    # The CALL, not the word: the comment above it names torch's check in order
    # to explain why it is not used, and an earlier version of this test matched
    # that prose and failed.
    assert "torch.cuda.is_bf16_supported()" not in source


def test_cpu_reports_no_bf16():
    import torch

    from src.model import setup

    if not torch.cuda.is_available():
        assert setup.supports_bf16() is False
        assert setup.dtype() is torch.float32


# --- the learning rate must suit the method -----------------------------------

def test_the_whole_lr_sweep_is_accepted():
    """Every rate the sweep will actually try must pass, or the guard would
    reject the experiment it exists to protect."""
    from src.model import setup

    for lr in config.LR_SWEEP:
        setup.assert_matched_learning_rate("lora", lr)
    setup.assert_matched_learning_rate("lora", config.LEARNING_RATE)


def test_a_full_finetuning_rate_is_rejected():
    """The quietest configuration error in the sweep. Only ~0.9% of weights are
    trainable and the B side starts at zero, so 2e-5 barely moves the adapter —
    the run completes with a nearly flat loss curve, which looks exactly like
    'fine-tuning did not help', this project's own H2 prediction."""
    from src.model import setup

    try:
        setup.assert_matched_learning_rate("lora", 2e-5)
    except AssertionError as error:
        assert "flat loss curve" in str(error)
        return
    raise AssertionError("a full fine-tuning learning rate was accepted for LoRA")


def test_a_diverging_rate_is_rejected():
    from src.model import setup

    try:
        setup.assert_matched_learning_rate("lora", 0.1)
    except AssertionError:
        return
    raise AssertionError("a diverging learning rate was accepted")


def test_run_one_can_actually_be_called():
    """The dangling reference this replaces: run_one called
    setup.assert_matched_learning_rate, which did not exist. Tests exercised
    plan() and final_specs() but never run_one's body, so the sweep failed on
    its first real run instead of in CI."""
    import inspect

    from src.model import setup
    from src.train import sweep

    source = inspect.getsource(sweep.run_one)
    for name in {line.split("setup.")[1].split("(")[0]
                 for line in source.splitlines() if "setup." in line}:
        assert hasattr(setup, name), f"sweep.run_one calls missing setup.{name}"
