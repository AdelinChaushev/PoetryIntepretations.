"""Train one LoRA adapter with TRL's ``SFTTrainer``.

Runs on Kaggle. Three things must be produced **in the same session as the
run**, because only the adapter comes back down and the model never does:

* the adapter itself
* a row in ``results/runs.csv``
* **validation perplexity**, which H4 needs

Forgetting the third costs a whole GPU session to repair.

**TRL's ``SFTTrainer`` does the loop and the masking.** Gradient accumulation,
the warmup and cosine schedule, gradient clipping, checkpointing, mixed
precision, early stopping, best-weight restoration — and, with
``completion_only_loss=True`` on a prompt-completion dataset, the label masking
itself. Verified rather than assumed: on a worked example it masks exactly the
prompt's token count and supervises exactly the completion plus EOS.

Two earlier versions of this module hand-rolled first the loop and then the
masking. The loop version shipped a bug that silently disabled early stopping
across a whole sweep, which is the argument for the library stated as evidence
rather than as preference.

What stays custom is one thing the library has no way to know about: an
**author-grouped validation split**, because a random one would put a poet on
both sides and report a validation loss for an author the model has seen.

**The budget is an epoch ceiling, and early stopping decides the real length.**
Neither pure scheme works: fixed steps gives equal updates but unequal
repetition, fixed epochs gives equal repetition but unequal updates. So the
ceiling is stated in epochs, clamped by a step floor, and each run stops on its
own validation curve.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path

import config
from src.train import dataset

log = logging.getLogger(__name__)


def perplexity(loss: float) -> float:
    """``exp`` of a masked cross-entropy loss.

    Because the loss is masked, this is the perplexity of **the interpretation
    given the poem** — not of the poem, and not of the pair. H4 compares it
    against judge scores.

    Read it as an effective branching factor: perplexity 20 means the model was
    about as uncertain as if choosing uniformly among 20 tokens. The floor is 1
    (``exp(0)``), and the ceiling for an untrained model is the vocabulary size,
    since uniform probability over ``|V|`` tokens gives ``log|V|``.

    Clamped because ``exp`` diverges fast: a run that blew up would otherwise
    write an unusable number into ``runs.csv``, or overflow outright. Anything
    at the clamp has failed, and the exact value carries no information.
    """
    return math.exp(min(loss, config.MAX_LOG_PERPLEXITY))


def split_validation(examples: list[dict], pairs: list[dict],
                     fraction: float = 0.1, seed: int | None = None):
    """Hold back a validation slice, **grouped by author**.

    Custom because no library does it: a random 10% would put Dickinson on both
    sides and report a validation loss for a poet the model has already seen.
    Since H4 compares validation perplexity against judge scores, an optimistic
    perplexity distorts exactly the ranking under test.

    Carved from the TRAINING partition, never the held-out fold — using that
    would mean selecting on the data results are later reported from.
    """
    rng = random.Random(config.SEED if seed is None else seed)
    author_of = {pair["poem_id"]: pair["author"] for pair in pairs}

    authors = sorted({author_of[e["poem_id"]] for e in examples
                      if e["poem_id"] in author_of})
    rng.shuffle(authors)

    per_author = {}
    for example in examples:
        per_author.setdefault(author_of.get(example["poem_id"]), []).append(example)

    # Authors arrive in whole blocks, so a naive "add until we reach 10%" can
    # overshoot badly when one author holds a large share — on a small fold that
    # produced a 6/35 split, i.e. more validation than training. Authors are
    # taken smallest-first and skipped when they would push past the ceiling,
    # so the overshoot is bounded by one author rather than unbounded.
    target = max(1, int(len(examples) * fraction))
    ceiling = max(target, int(len(examples) * config.VALIDATION_MAX_FRACTION))

    held, count = set(), 0
    for author in sorted(authors, key=lambda a: len(per_author.get(a, []))):
        size = len(per_author.get(author, []))
        if count >= target or count + size > ceiling:
            continue
        held.add(author)
        count += size

    train = [e for e in examples if author_of.get(e["poem_id"]) not in held]
    validation = [e for e in examples if author_of.get(e["poem_id"]) in held]

    assert train, (
        f"the author-grouped validation split left NO training examples: "
        f"{len(examples)} examples across {len(authors)} authors. Too few "
        f"authors to hold any back without taking everything.")
    log.info("validation split: %d train / %d validation (%d authors held back)",
             len(train), len(validation), len(held))
    return train, validation


def training_arguments(run_name: str, steps: int, output_dir, **overrides):
    """Map ``config`` onto ``SFTConfig``.

    One place, so a setting cannot be applied at one call site and forgotten at
    another. This transformers version has no ``warmup_ratio``, so the ratio is
    converted here rather than a step count being hardcoded — otherwise warmup
    would silently change meaning whenever the budget moved.
    """
    import torch
    from trl import SFTConfig

    use_cuda = torch.cuda.is_available()
    return SFTConfig(
        output_dir=str(output_dir),
        max_steps=steps,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        learning_rate=overrides.get("learning_rate", config.LEARNING_RATE),
        weight_decay=config.WEIGHT_DECAY,
        lr_scheduler_type=config.LR_SCHEDULE,
        warmup_steps=max(1, int(steps * config.WARMUP_RATIO)),
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=config.EVAL_EVERY_STEPS,
        logging_steps=config.EVAL_EVERY_STEPS,
        save_strategy="steps",
        save_steps=config.EVAL_EVERY_STEPS,
        # Early stopping is worthless without this: a run ends `patience`
        # evaluations past its own best, so keeping the last state would save
        # the overfitted model it stopped because of.
        load_best_model_at_end=config.RESTORE_BEST_WEIGHTS,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        # THE masking setting. With a prompt-completion dataset this supervises
        # the completion only, so the poem is read but never scored. Stated
        # explicitly rather than left to the default, which infers it from the
        # dataset shape — an inferred behaviour is one a schema change could
        # silently flip.
        completion_only_loss=True,
        # Nothing reaches here that does not already fit: dataset.build_dataset
        # DROPS over-long pairs rather than letting them be truncated, because a
        # truncated poem is still scored against its full text by the grounding
        # checker.
        max_length=config.MAX_SEQ_LEN,
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        fp16=use_cuda and not torch.cuda.is_bf16_supported(),
        seed=config.SEED,
        data_seed=config.SEED,
        report_to=[],
    )


def train(model, tokenizer, examples: list[dict], pairs: list[dict],
          run_name: str, max_steps: int | None = None,
          early_stopping: bool | None = None, **overrides) -> dict:
    """Train one adapter. Returns the run record written to ``runs.csv``."""
    from datasets import Dataset
    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer

    stopping = config.EARLY_STOPPING if early_stopping is None else early_stopping
    train_set, validation = split_validation(examples, pairs)
    steps = max_steps or config.steps_for(len(train_set))
    effective = config.BATCH_SIZE * config.GRAD_ACCUM_STEPS

    log.info("budget: %d steps = %.1f epochs over %d examples "
             "(effective batch %d, early stopping %s)",
             steps, steps * effective / max(1, len(train_set)), len(train_set),
             effective, "on" if stopping else "off")

    trainer = SFTTrainer(
        model=model,
        args=training_arguments(run_name, steps,
                                config.RESULTS_DIR / "checkpoints" / run_name,
                                **overrides),
        train_dataset=Dataset.from_list(train_set),
        eval_dataset=Dataset.from_list(validation),
        processing_class=tokenizer,
        callbacks=([EarlyStoppingCallback(
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE)]
            if stopping else []),
    )

    started = time.time()
    result = trainer.train()
    final = trainer.evaluate()
    save_history(trainer.state.log_history, run_name)

    val_loss = final["eval_loss"]
    record = {
        "run": run_name,
        "model": config.MODEL,
        "rank": overrides.get("rank", config.LORA_RANK),
        "alpha": config.lora_alpha(overrides.get("rank", config.LORA_RANK)),
        "target_modules": ",".join(config.LORA_TARGET_MODULES),
        "learning_rate": overrides.get("learning_rate", config.LEARNING_RATE),
        "batch_size": config.BATCH_SIZE,
        "grad_accum": config.GRAD_ACCUM_STEPS,
        "effective_batch": effective,
        "max_steps": steps,
        "steps_run": int(trainer.state.global_step),
        "epochs_run": round(trainer.state.global_step * effective
                            / max(1, len(train_set)), 2),
        "early_stopped": int(trainer.state.global_step) < steps,
        "early_stopping_enabled": stopping,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "n_train": len(train_set),
        "n_validation": len(validation),
        "final_train_loss": result.training_loss,
        "final_val_loss": val_loss,
        # Needs the model, and the model stays on Kaggle. Computing it later
        # costs a whole GPU session.
        "val_perplexity": perplexity(val_loss),
        "trainable_params": sum(p.numel() for p in model.parameters()
                                if p.requires_grad),
        "seed": config.SEED,
        "wall_clock_seconds": round(time.time() - started, 1),
        # So a smoke row is identifiable even if it reaches the real file by
        # some other route. sweep.select_winner refuses to select on it.
        "smoke": config.SMOKE,
    }
    append_run(record)
    return record


def save_history(history: list[dict], run_name: str) -> None:
    """Trainer's log history, for the loss-curve figure."""
    path = config.RESULTS_DIR / "histories"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{run_name}.json").write_text(json.dumps(history, indent=2))


def append_run(record: dict) -> None:
    """Append one row to ``runs.csv``, creating it with a header if absent.

    Append rather than rewrite: a Kaggle session that dies mid-sweep must not
    take the earlier rows with it.
    """
    import csv

    path = Path(config.RUNS_CSV_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        if not exists:
            writer.writeheader()
        writer.writerow(record)
    log.info("run recorded in %s", path)


def _smoke() -> None:
    """End-to-end on CPU in under a minute, so a bug surfaces here.

    This module and ``generate.inference`` are the only two that never
    otherwise run locally, which makes them the two where a bug costs a Kaggle
    session to find. Under ``SMOKE`` the model, the step count and the corpus
    all shrink, and the run record goes to ``runs_smoke.csv`` rather than the
    real one.
    """
    from src.data import splits
    from src.model import setup
    from src.train import dataset

    config.configure_logging()
    assert config.SMOKE, "run with SMOKE=1; this is a size flag, not a mode"

    pairs = splits.load_training_pairs()[:config.N_POEMS or 20]
    assert pairs, f"no training pairs at {config.TRAINING_PAIRS_PATH}"

    tokenizer = setup.load_tokenizer()
    model = setup.apply_lora(setup.load_base_model())
    examples = dataset.build_dataset(pairs, tokenizer)
    assert examples, (
        f"every pair exceeded MAX_SEQ_LEN={config.MAX_SEQ_LEN}; nothing to "
        f"train on")

    record = train(model, tokenizer, examples, pairs, run_name="smoke",
                   max_steps=config.MAX_STEPS)
    print(f"\n  {record['steps_run']} steps on {record['n_train']} examples "
          f"in {record['wall_clock_seconds']}s")
    print(f"  val loss {record['final_val_loss']:.4f}  ->  perplexity "
          f"{record['val_perplexity']:.2f}")
    print(f"  written to {config.RUNS_CSV_PATH.name} (smoke={record['smoke']})")


if __name__ == "__main__":
    _smoke()
