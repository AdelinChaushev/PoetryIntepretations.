"""Load the student and attach LoRA adapters.

Runs on Kaggle. Everything here is guarded so the module can be imported — and
smoke-tested — on a laptop with no GPU, because the two modules that never
otherwise run locally are the two where a bug costs a GPU session to discover.

**The base model is frozen and that is asserted, not assumed.** ``peft`` freezes
it for us, but a mis-specified ``target_modules`` matches nothing, trains zero
parameters, and produces a run that completes with a flat loss curve and an
adapter full of zeros. :func:`assert_adapters_only` turns both failures into
errors.
"""

from __future__ import annotations

import logging

import config

log = logging.getLogger(__name__)


def device() -> str:
    """Where to put the model. Never assumes CUDA at import time."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch absent on the laptop
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def supports_bf16() -> bool:
    """True only on hardware ``transformers`` will ACCEPT for bf16 training.

    Not ``torch.cuda.is_bf16_supported()``, which returns **True on Turing**
    because it counts emulation. ``TrainingArguments`` applies a stricter rule —
    Ampere or newer — and raises "Your setup doesn't support bf16/gpu".

    Using the loose check to pick the model dtype while transformers applies the
    strict one to the trainer is how a model gets loaded in bf16 and then
    refused by the trainer meant to train it. Observed on a Kaggle T4: the model
    loaded as bfloat16 and SFTConfig would not construct.

    Compute capability 8.0 is Ampere. T4 is 7.5 and P100 is 6.0 — the two cards
    this project targets — so this is False on both and everything downstream
    agrees on fp16.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def dtype():
    """bf16 on hardware that supports it, fp16 on older cards, fp32 on CPU.

    T4 (Turing) and P100 (Pascal) predate bf16, so this cannot be hardcoded —
    requesting bf16 there is either an error or a silent fallback that halves
    throughput.
    """
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if supports_bf16() else torch.float16


def load_tokenizer():
    """Load the student's tokeniser, with a pad token guaranteed.

    Qwen and GPT-2 both ship without a distinct pad token. Left unset, the
    collator has nothing to pad with; set to EOS without care, the model would
    be unable to tell padding from a genuine end of sequence. Padding is masked
    out of the loss and out of attention either way, so reusing EOS is safe —
    but it has to be deliberate rather than a default.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("pad_token was unset; using eos_token (%s). Padding is masked "
                 "in labels and attention, so this cannot leak into the loss.",
                 tokenizer.eos_token_id)
    return tokenizer


def load_base_model():
    """Load the frozen student.

    Base, not Instruct — an instruction-tuned checkpoint would already follow
    the output schema, which would contaminate the format-compliance
    measurement that H1 is about.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL,
        dtype=dtype(),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    log.info("loaded %s (%s params) on %s as %s", config.MODEL,
             f"{sum(p.numel() for p in model.parameters()):,}", device(), dtype())
    return model.to(device())


def lora_config(rank: int | None = None, target_modules=None, dropout=None):
    """Build the peft config for a given rank.

    ``alpha`` is DERIVED from the rank rather than fixed, so the ``alpha / r``
    scaling stays constant across the rank sweep. With alpha fixed, sweeping
    {4, 8, 16} would vary the scaling 4x, 2x, 1x and the axis would measure
    capacity and effective step size together.
    """
    from peft import LoraConfig

    rank = config.LORA_RANK if rank is None else rank
    # A float rank reaches nn.Linear(in_features, r) and fails several frames
    # inside peft with a message naming neither the rank nor its source. It
    # arrives that way from runs.csv, where every column round-trips as text.
    assert isinstance(rank, int) and not isinstance(rank, bool), (
        f"rank must be an int, got {rank!r} ({type(rank).__name__}). peft builds "
        f"nn.Linear(in_features, r) and a float fails deep inside it.")
    return LoraConfig(
        r=rank,
        lora_alpha=config.lora_alpha(rank),
        lora_dropout=config.LORA_DROPOUT if dropout is None else dropout,
        target_modules=list(target_modules or config.LORA_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )


def apply_lora(model, rank: int | None = None, target_modules=None):
    """Attach adapters and verify that only they are trainable."""
    from peft import get_peft_model

    adapted = get_peft_model(model, lora_config(rank, target_modules))
    assert_adapters_only(adapted)
    log.info("LoRA r=%d alpha=%d on %s", rank or config.LORA_RANK,
             config.lora_alpha(rank or config.LORA_RANK),
             list(target_modules or config.LORA_TARGET_MODULES))
    return adapted


def trainable_parameters(model) -> dict:
    """Trainable and total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total,
            "percent": 100 * trainable / total if total else 0.0}


def assert_adapters_only(model) -> None:
    """Only LoRA parameters may be trainable, and there must be some.

    Two failures, both silent. A mis-specified ``target_modules`` matches
    nothing: the run completes, the loss curve is flat, and the saved adapter
    is empty — which looks like "fine-tuning did not help", the project's own
    H2 prediction. And a base weight left unfrozen would make the run a partial
    full fine-tune reported as LoRA.
    """
    counts = trainable_parameters(model)
    assert counts["trainable"] > 0, (
        f"no trainable parameters — target_modules "
        f"{list(config.LORA_TARGET_MODULES)} matched nothing in {config.MODEL}. "
        f"The run would complete with a flat loss curve and an empty adapter, "
        f"which is indistinguishable from 'fine-tuning did not help'."
    )
    leaked = [name for name, param in model.named_parameters()
              if param.requires_grad and "lora_" not in name]
    assert not leaked, (
        f"{len(leaked)} non-adapter parameters are trainable, starting with "
        f"{leaked[:3]} — this is a partial full fine-tune, not LoRA"
    )
    log.info("trainable %s / %s (%.2f%%) — adapters only",
             f"{counts['trainable']:,}", f"{counts['total']:,}",
             counts["percent"])


def save_adapter(model, path) -> None:
    """Write the adapter, which records its own base model in the config.

    Never the merged model: merged into the base it is ~1 GB in fp16, past
    GitHub's file limit, and reconstructible from these two files in one line.
    """
    from pathlib import Path

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    size = sum(f.stat().st_size for f in path.glob("*")) / 1024**2
    log.info("adapter written to %s (%.1f MB)", path, size)


def load_adapter(path, base=None):
    """Attach a saved adapter to a base model.

    ``adapter_config.json`` records the base model it was trained against, so
    the adapter is self-describing and this needs no rank or target list. That
    is also why merged models are never saved: the base plus these two files
    reconstructs the model in one line, at 1.5 MB instead of a gigabyte.
    """
    from peft import PeftModel

    model = load_base_model() if base is None else base
    adapted = PeftModel.from_pretrained(model, str(path))
    log.info("loaded adapter %s onto %s", path, config.MODEL)
    return adapted


#: LoRA needs a learning rate one to two orders of magnitude above full
#: fine-tuning's. Only ~0.9% of the weights are trainable, they start at zero on
#: the B side, and the update is scaled by alpha/r — so a rate copied from a
#: full fine-tuning recipe (1e-5 to 5e-5) barely moves the adapter and produces
#: a run that completes with a nearly flat loss curve.
LORA_LEARNING_RATE_RANGE: tuple[float, float] = (5e-5, 1e-3)


def assert_matched_learning_rate(method: str, learning_rate: float) -> None:
    """The learning rate must suit the adaptation method.

    A guard against the quietest configuration error in the sweep: a full
    fine-tuning learning rate applied to LoRA. Nothing raises, the run completes,
    the loss curve is nearly flat — and "fine-tuning did not help" is this
    project's own H2 prediction, so the result looks like a finding rather than
    a mistake.
    """
    if method != "lora":
        return

    low, high = LORA_LEARNING_RATE_RANGE
    assert low <= learning_rate <= high, (
        f"learning rate {learning_rate:g} is outside the range LoRA needs "
        f"({low:g} to {high:g}). Below it the adapter barely moves and the run "
        f"finishes with a flat loss curve, which is indistinguishable from "
        f"'fine-tuning did not help' — the very thing H2 predicts. Above it the "
        f"adapter diverges.")
