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

    from src.model import setup

    use_cuda = torch.cuda.is_available()
    # setup.supports_bf16, not torch.cuda.is_bf16_supported: the torch check
    # counts emulation and returns True on Turing, while TrainingArguments
    # requires Ampere and raises. Routed through the same helper that picks the
    # model dtype, so the weights and the trainer cannot disagree.
    bf16 = setup.supports_bf16()
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
        # THE masking setting. With a prompt-completion dataset, True supervises
        # the completion only, so the poem is read but never scored. Stated
        # explicitly rather than left to the default, which infers it from the
        # dataset shape — an inferred behaviour is one a schema change could
        # silently flip.
        #
        # Driven by the `masking` override so the sweep's unmasked run actually
        # trains unmasked. It previously did not: the spec carried
        # masking="unmasked", run_name labelled the row `_unmasked`, and this
        # line hardcoded True — so the run trained masked and the report would
        # have shown a masking comparison in which nothing varied.
        completion_only_loss=overrides.get("masking", "masked") == "masked",
        # Nothing reaches here that does not already fit: dataset.build_dataset
        # DROPS over-long pairs rather than letting them be truncated, because a
        # truncated poem is still scored against its full text by the grounding
        # checker.
        max_length=config.MAX_SEQ_LEN,
        bf16=bf16,
        fp16=use_cuda and not bf16,
        seed=config.SEED,
        data_seed=config.SEED,
        report_to=[],
    )


def heldout_examples(pairs: list[dict], fold: int, tokenizer) -> list[dict]:
    """The poems ``fold`` holds out, as training-shaped examples.

    Used for a perplexity that model selection never touched. Not the same set
    as the validation slice, and deliberately so — see :func:`train`.
    """
    from src.data import splits

    mapping = splits.load_assignment()
    held = [pair for pair in pairs if mapping.get(pair["poem_id"]) == fold]
    return dataset.build_dataset(held, tokenizer)


def evaluate_perplexity(model, tokenizer, examples: list[dict],
                        label: str = "eval",
                        max_length: int | None = None) -> dict:
    """Masked cross-entropy and perplexity of ``model`` on ``examples``.

    A forward pass only — no gradients, no optimizer, nothing written. Shared by
    the held-out measurement inside :func:`train` and by
    :func:`base_perplexity`, so an adapted model and the base model are always
    scored by the same code on the same masking.

    **Its own config, not ``training_arguments``.** Two settings must differ.
    ``max_length`` defaults to ``GEN_MAX_CONTEXT`` rather than ``MAX_SEQ_LEN``,
    because ``base_few``'s sequences carry three exemplars and reach ~3,900
    tokens — inheriting the training cap would truncate away the target poem and
    report the perplexity of an interpretation of a poem the model never saw.
    And the batch is 1: sequences twice training's length would otherwise
    multiply the quadratic attention term for no benefit, since nothing here is
    being optimised.
    """
    import tempfile

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    from src.model import setup

    if not examples:
        return {"loss": float("nan"), "perplexity": float("nan"), "n": 0}

    limit = config.GEN_MAX_CONTEXT if max_length is None else max_length
    too_long = [e for e in examples
                if dataset.token_length(e, tokenizer) + 1 > limit]
    assert not too_long, (
        f"{len(too_long)} of {len(examples)} examples exceed {limit} tokens and "
        f"would be TRUNCATED, which for base_few removes the target poem. Raise "
        f"the limit rather than measuring a truncated sequence.")

    bf16 = setup.supports_bf16()
    with tempfile.TemporaryDirectory() as tmp:
        trainer = SFTTrainer(
            model=model,
            args=SFTConfig(output_dir=tmp, report_to=[], eval_strategy="no",
                           completion_only_loss=True, max_length=limit,
                           per_device_eval_batch_size=1,
                           per_device_train_batch_size=1, max_steps=1,
                           bf16=bf16,
                           fp16=not bf16 and setup.device() == "cuda",
                           seed=config.SEED),
            train_dataset=Dataset.from_list(examples[:1]),
            eval_dataset=Dataset.from_list(examples),
            processing_class=tokenizer,
        )
        metrics = trainer.evaluate()

    loss = metrics["eval_loss"]
    return {"loss": loss, "perplexity": perplexity(loss), "n": len(examples)}


def few_shot_examples(pairs: list[dict], fold: int, tokenizer,
                      exemplars: list[dict]) -> list[dict]:
    """Held-out examples under ``base_few``'s own prompt.

    ``base_few`` differs from ``base_zero`` only in what precedes the poem, so
    measuring both under the zero-shot prompt would give them an identical
    perplexity and make them indistinguishable to H4 by construction. Scored in
    the context the arm actually generates in instead.
    """
    from src.data import splits
    from src.generate import inference

    mapping = splits.load_assignment()
    return [{"prompt": inference.build_prompt(pair, "base_few", exemplars),
             "completion": pair["interpretation"],
             "poem_id": pair["poem_id"]}
            for pair in pairs if mapping.get(pair["poem_id"]) == fold]


def base_perplexity(pairs: list[dict], model, tokenizer,
                    exemplars: list[dict] | None = None) -> dict:
    """The BASE model's held-out perplexity, per arm and per fold.

    ``base_zero`` and ``base_few`` need a perplexity for H4, and the base model
    has no held-out fold of its own — it never trained, so no poem is off-limits
    to it. That freedom is the problem: measured on a different set from the
    LoRA arms, the number would not be comparable, and H4 correlates perplexity
    against judge score **across arms**.

    So it is measured on exactly the folds the adapters are measured on, and
    each arm is measured **under its own prompt**. Same weights, different
    context: without that, the two base arms would return the same number and
    H4 would be comparing a tie it created itself.

    Must run on Kaggle, in the same session: it needs the base model, and only
    adapters come down.
    """
    arms = {"base_zero": lambda fold: heldout_examples(pairs, fold, tokenizer)}
    if exemplars:
        arms["base_few"] = lambda fold: few_shot_examples(pairs, fold,
                                                          tokenizer, exemplars)

    result: dict = {}
    for arm, build in arms.items():
        per_fold = {}
        for fold in range(config.N_FOLDS):
            examples = build(fold)
            per_fold[fold] = evaluate_perplexity(
                model, tokenizer, examples, label=f"{arm}_fold{fold}")
            log.info("%s, fold %d held out: perplexity %.3f over %d examples",
                     arm, fold, per_fold[fold]["perplexity"], per_fold[fold]["n"])

        losses = [m["loss"] for m in per_fold.values() if m["n"]]
        pooled = sum(losses) / len(losses) if losses else float("nan")
        result[arm] = {"per_fold": per_fold, "mean_loss": pooled,
                       "mean_perplexity": perplexity(pooled) if losses
                                          else float("nan")}

    # Both arms must cover the same poems, or the comparison is between
    # different data rather than between different prompts.
    if len(result) > 1:
        counts = {arm: [m["n"] for m in r["per_fold"].values()]
                  for arm, r in result.items()}
        assert len(set(map(tuple, counts.values()))) == 1, (
            f"the base arms cover different poems per fold: {counts}. One "
            f"prompt is dropping examples the other keeps, so the perplexities "
            f"are not comparable.")
    return result


def train(model, tokenizer, examples: list[dict], pairs: list[dict],
          run_name: str, max_steps: int | None = None,
          early_stopping: bool | None = None, fold: int | None = None,
          save_adapter: bool = True, **overrides) -> dict:
    """Train one adapter. Returns the run record written to ``runs.csv``.

    Pass ``fold`` to have the adapter written here, in the same call that
    trained it. Kaggle sessions die, and an adapter saved by a *later* notebook
    cell is one a dead kernel loses — costing the whole run, not just the cell.
    The path comes from :func:`config.adapter_dir`, the same function
    ``inference.adapter_for`` reads back, so the two cannot drift.

    With ``load_best_model_at_end`` the object saved holds the **best** weights
    rather than the ones the run stopped on. Verified rather than assumed: the
    written adapter is byte-identical to the best checkpoint and differs from
    the last.

    ``save_adapter=False`` for sweep runs. Nine grid points at three ranks would
    otherwise collide — every rank-8 configuration writes to the same
    ``adapter_dir(8, fold)``, so the last one silently overwrites the rest and
    the surviving adapter belongs to no recorded run in particular.

    **Two perplexities are recorded, and the difference between them matters.**
    ``val_perplexity`` comes from the validation slice, which early stopping
    *selected on* — the run chose its stopping point by minimising exactly that
    number, so it is optimistic. ``heldout_perplexity`` comes from the fold this
    run never saw at all. H4 should use the second: it correlates perplexity
    against judge score across arms, and the base arms' perplexity was never
    selected on, so using the selected-on number for the LoRA arms alone would
    bias the comparison in their favour.
    """
    from datasets import Dataset
    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer

    import torch

    # Fail here, clearly, rather than inside a DataParallel worker thread.
    # With >1 visible GPU, Trainer wraps the model in nn.DataParallel, which
    # replicates it per batch and breaks under PEFT -- the replica on cuda:1
    # gets input indices while embed_tokens keeps its weights on cuda:0. The
    # resulting error names neither DataParallel nor the device count, and
    # arrives only once training has started.
    # Training on CPU is not an error anywhere in torch -- it is a warning at
    # most ("no accelerator is found") and then it simply runs, roughly two
    # orders of magnitude slower. On Kaggle that means a session spent producing
    # nothing, discovered only when the clock runs out. SMOKE is exempt: it is a
    # size flag and is meant to run on a laptop.
    assert torch.cuda.is_available() or config.SMOKE, (
        "no GPU is visible, so this run would train on CPU -- silently, and far "
        "too slowly to finish. Check Settings -> Accelerator in the Kaggle "
        "sidebar (a session restart can drop it, and so can the weekly quota). "
        "Set SMOKE=1 if a tiny CPU run is what you actually want.")

    visible = torch.cuda.device_count()
    assert visible <= 1, (
        f"{visible} GPUs are visible, so Trainer will use nn.DataParallel and "
        f"fail under PEFT with a device mismatch. Set CUDA_VISIBLE_DEVICES=0 "
        f"BEFORE torch initialises CUDA (the first notebook cell), or choose a "
        f"single-GPU accelerator. Nothing is lost: a 0.5B model with LoRA fits "
        f"on one card and the batch is too small to split usefully.")

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
        # threshold, not just patience: without it any improvement counts,
        # including 0.0001, so a creeping run resets the counter forever and
        # spends the whole budget.
        callbacks=([EarlyStoppingCallback(
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=config.EARLY_STOPPING_THRESHOLD)]
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
        "fold": fold,
        "adapter": None,
        "heldout_loss": None,
        "heldout_perplexity": None,
        "n_heldout": 0,
    }

    if fold is not None:
        # The perplexity model selection never touched. Measured AFTER
        # load_best_model_at_end has restored the best weights, so it describes
        # the adapter that actually ships rather than the one the run stopped on.
        held = heldout_examples(pairs, fold, tokenizer)
        metrics = evaluate_perplexity(trainer.model, tokenizer, held,
                                      label=f"{run_name}_heldout")
        record.update(heldout_loss=metrics["loss"],
                      heldout_perplexity=metrics["perplexity"],
                      n_heldout=metrics["n"])
        log.info("fold %d held out: perplexity %.3f over %d examples "
                 "(validation said %.3f, but selection chose on it)",
                 fold, metrics["perplexity"], metrics["n"],
                 record["val_perplexity"])

    if save_adapter:
        from src.model import setup

        # trainer.model IS the object passed in, and load_best_model_at_end has
        # already restored the best weights into it, so this writes the best
        # adapter rather than the one the run happened to stop on.
        #
        # Keyed on the run name, not (rank, fold): three learning rates at rank
        # 8 share one (rank, fold) and would overwrite each other. Every run
        # keeps its weights, so every row in runs.csv is traceable to the model
        # that produced it.
        path = config.run_adapter_dir(run_name)
        setup.save_adapter(trainer.model, path)
        record["adapter"] = str(path)

    append_run(record)
    return record


def update_run(run_name: str, **fields) -> bool:
    """Rewrite one recorded run's fields in place. Returns whether it matched.

    ``append_run`` only appends, which is right for a live run — a session that
    dies mid-sweep must not take earlier rows with it. Backfilling a column is
    the one case that genuinely needs a rewrite, and doing it through the same
    header reconciliation keeps the file consistent either way.
    """
    import csv

    path = Path(config.RUNS_CSV_PATH)
    if not path.exists():
        return False

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header, rows = list(reader.fieldnames or []), list(reader)

    matched = False
    for row in rows:
        if row.get("run") == run_name:
            row.update({k: v for k, v in fields.items()})
            matched = True
    if not matched:
        return False

    fields_out = header + [f for f in fields if f not in header]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_out,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields_out})
    log.info("updated %s in %s", run_name, path.name)
    return True


def backfill_heldout_perplexity(pairs: list[dict], tokenizer) -> list[dict]:
    """Compute ``heldout_perplexity`` for recorded runs that lack it.

    **Much cheaper than re-running, and more faithful.** The adapter is already
    on disk and the fold it held out is already in the row, so this is one
    forward pass over ~500 poems rather than a whole training run. It also
    measures the weights that actually shipped, where a retrain would produce a
    *different* adapter — same configuration, but GPU non-determinism means not
    the same numbers.

    Exists because the day-3 pilot predates the column: it is a valid fold-0
    run at the default configuration, missing only this measurement.
    """
    import csv

    from src.model import setup

    path = Path(config.RUNS_CSV_PATH)
    if not path.exists():
        return []

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    filled = []
    for row in rows:
        if row.get("heldout_perplexity") not in (None, "", "None"):
            continue
        adapter, fold = row.get("adapter"), row.get("fold")
        if not adapter or fold in (None, "", "None"):
            continue
        if not Path(adapter).exists():
            log.warning("%s records adapter %s but it is not on disk",
                        row["run"], adapter)
            continue

        model = setup.load_adapter(adapter)
        held = heldout_examples(pairs, int(fold), tokenizer)
        metrics = evaluate_perplexity(model, tokenizer, held,
                                      label=f"{row['run']}_backfill")
        update_run(row["run"], heldout_loss=metrics["loss"],
                   heldout_perplexity=metrics["perplexity"],
                   n_heldout=metrics["n"])
        log.info("%s: held-out perplexity %.3f over %d examples",
                 row["run"], metrics["perplexity"], metrics["n"])
        filled.append({"run": row["run"], **metrics})

    if not filled:
        log.info("no runs needed backfilling")
    return filled


#: Everything a GPU session produces that cannot be rebuilt on the laptop,
#: because only adapters come down and the model never does.
SESSION_ARTIFACTS: tuple[str, ...] = ("runs.csv", "adapters", "histories",
                                      "contamination.jsonl",
                                      "base_perplexity.json")


def archive_results(destination=None) -> Path:
    """Bundle this session's output into one zip to download.

    ``/kaggle/working`` is wiped when a session ends, and ``runs.csv`` is the
    only record that a run happened — ``sweep.load_completed`` reads it to decide
    what to skip. A resume file that lives only in the directory Kaggle deletes
    is not a resume file, and twelve runs were lost proving it.

    Safe to call at any point: missing pieces are reported and skipped rather
    than raising, so it works mid-sweep as well as at the end.
    """
    import shutil

    config.ensure_dirs()
    stage = Path(config.RESULTS_DIR) / "to_download"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)

    for name in SESSION_ARTIFACTS:
        source = Path(config.RESULTS_DIR) / name
        if not source.exists():
            log.info("  %-22s absent, skipped", name)
            continue
        if source.is_dir():
            shutil.copytree(source, stage / name)
            size = sum(f.stat().st_size for f in source.rglob("*") if f.is_file())
        else:
            shutil.copy2(source, stage / name)
            size = source.stat().st_size
        log.info("  %-22s %.2f MB", name, size / 1024 ** 2)

    target = Path(config.RESULTS_DIR) / "session_output" if destination is None \
        else Path(destination)
    archive = Path(shutil.make_archive(str(target), "zip", stage))
    log.info("wrote %s (%.2f MB) — DOWNLOAD IT before the session ends",
             archive, archive.stat().st_size / 1024 ** 2)
    return archive


def restore_results(source) -> dict:
    """Copy a previous session's artifacts back into place.

    The half that was missing. ``sweep.load_completed`` reads ``runs.csv`` from
    ``RESULTS_DIR``; without this, every new session starts from an empty file
    and re-runs everything already paid for. Point ``source`` at the unzipped
    Kaggle Dataset holding a previous ``session_output``.

    Adapters are restored too, so the six final runs resume properly rather than
    retraining folds whose weights already exist.
    """
    import shutil

    source = Path(source)
    assert source.exists(), (
        f"{source} does not exist. Upload the session_output zip as a Kaggle "
        f"Dataset and point this at its directory, or the sweep starts over.")

    config.ensure_dirs()
    restored = {}
    for name in SESSION_ARTIFACTS:
        found = source / name
        if not found.exists():
            continue
        target = Path(config.RESULTS_DIR) / name
        if found.is_dir():
            shutil.copytree(found, target, dirs_exist_ok=True)
            restored[name] = len(list(found.iterdir()))
        else:
            shutil.copy2(found, target)
            restored[name] = 1
        log.info("restored %s", name)

    if not restored:
        log.warning("nothing restored from %s — check the path. Every run will "
                    "be repeated.", source)
    return restored


def save_history(history: list[dict], run_name: str) -> None:
    """Trainer's log history, for the loss-curve figure."""
    path = config.RESULTS_DIR / "histories"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{run_name}.json").write_text(json.dumps(history, indent=2))


def training_curve(run_name: str) -> dict:
    """A run's loss trajectory, with the post-restore evaluation removed.

    ``train`` calls ``trainer.evaluate()`` once more after
    ``load_best_model_at_end`` has restored the best weights, so the final entry
    in the log history **duplicates the best value** rather than continuing the
    trajectory. Plotted raw, the curve appears to recover at the end — loss
    climbing through overfitting and then dropping back — which is the weight
    restore, not learning.

    Returns train and eval series separately; they are logged at the same steps
    but the eval series has one extra entry, which is exactly the one to drop.
    """
    path = Path(config.RESULTS_DIR) / "histories" / f"{run_name}.json"
    if not path.exists():
        return {"train": [], "eval": [], "best": None}

    history = json.loads(path.read_text())
    train = [(e["epoch"], e["loss"]) for e in history
             if "loss" in e and "eval_loss" not in e]
    evals = [(e["epoch"], e["eval_loss"]) for e in history if "eval_loss" in e]

    # Drop the trailing restore evaluation: it repeats an epoch already present
    # and its value is the minimum by construction.
    if len(evals) > 1 and evals[-1][1] == min(v for _, v in evals):
        evals = evals[:-1]

    return {"train": train, "eval": evals,
            "best": min((v for _, v in evals), default=None)}


def append_run(record: dict) -> None:
    """Append one row to ``runs.csv``, migrating the header if it has changed.

    Append rather than rewrite: a Kaggle session that dies mid-sweep must not
    take the earlier rows with it. Each row is a GPU run, so losing one is
    expensive and corrupting one is worse.

    **The header is reconciled, not assumed.** A plain append writes no header
    when the file exists, so adding a field to the record puts N+1 values under
    N column names and every field after the new one shifts by a column —
    silently, in the file ``select_winner`` and every figure read from. That is
    not hypothetical: it happened here the first time a field was added
    mid-project. When the columns disagree the file is rewritten with the union,
    old rows padded with blanks, so no run is lost and none is misaligned.
    """
    import csv

    path = Path(config.RUNS_CSV_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: list[dict] = []
    header: list[str] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            existing_rows = list(reader)

    if not header:
        fields = list(record)
    elif set(header) == set(record):
        fields = header
    else:
        # Union, preserving the existing order so a reader's columns do not
        # shuffle, with genuinely new fields appended.
        fields = header + [f for f in record if f not in header]
        log.warning("runs.csv schema changed (%s); rewriting %d existing row(s) "
                    "with the union of columns rather than appending a "
                    "misaligned row",
                    ", ".join(f for f in fields if f not in header),
                    len(existing_rows))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields,
                                    extrasaction="ignore")
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({f: row.get(f, "") for f in fields})

    write_header = not path.exists() or not header
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({f: record.get(f, "") for f in fields})
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

    # fold is passed so the ADAPTER SAVE is exercised too. Without it the smoke
    # run verifies training and skips the step whose failure costs the run.
    record = train(model, tokenizer, examples, pairs, run_name="smoke",
                   max_steps=config.MAX_STEPS, fold=config.SINGLE_SPLIT_FOLD)
    print(f"\n  {record['steps_run']} steps on {record['n_train']} examples "
          f"in {record['wall_clock_seconds']}s")
    print(f"  val loss {record['final_val_loss']:.4f}  ->  perplexity "
          f"{record['val_perplexity']:.2f}")
    print(f"  written to {config.RUNS_CSV_PATH.name} (smoke={record['smoke']})")

    saved = Path(record["adapter"])
    print(f"\n  adapter {saved.name}")
    print(f"    files {sorted(f.name for f in saved.iterdir())}")
    print(f"    size  {sum(f.stat().st_size for f in saved.iterdir())/1024**2:.2f} MB")

    # It must load back onto a fresh base model, which is what the README
    # promises a reader and what generation does five times over.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(config.MODEL), str(saved))
    print(f"    reloads onto a fresh {config.MODEL}: True")

    from src.generate import inference
    expected = inference.adapter_for(1, "lora_r8", {1: config.SINGLE_SPLIT_FOLD})
    print(f"    generation looks for it here too: {saved == expected}")


if __name__ == "__main__":
    _smoke()
