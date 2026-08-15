"""Tests for the prompt-completion dataset and the masking TRL applies to it.

Masking is the highest-risk behaviour in the pipeline and it is now the
library's job, so these tests verify the LIBRARY masks where we expect rather
than verifying our own arithmetic. That is the more useful check: a TRL upgrade
that changed the boundary would otherwise be invisible until a result looked
strange.

The failure being guarded against is silent in every direction. Supervise the
prompt and the model learns to recite poems, which is easier than interpreting
them, so the loss *falls* while the model gets worse at the task.
"""

from __future__ import annotations

import pytest

import config
from src.train import dataset


def pair(interpretation: str = "It reads the tide as a figure for memory.",
         lines: list[str] | None = None) -> dict:
    return {"poem_id": 1, "title": "T", "author": "A",
            "lines": lines or ["first line here", "second line here"],
            "linecount": 2, "interpretation": interpretation}


@pytest.fixture(scope="module")
def tokenizer():
    from src.data.filter import get_tokenizer
    return get_tokenizer()


# --- the record shape ---------------------------------------------------------

def test_poem_lives_in_the_prompt_and_interpretation_in_the_completion():
    """This split is what tells TRL where the loss boundary is."""
    example = dataset.build_example(pair())
    assert "first line here" in example["prompt"]
    assert example["completion"] == pair()["interpretation"]


def test_prompt_matches_the_teacher_prompt():
    """A different prompt would make every interpretation a response to
    something the model was never shown."""
    from src.data import generate

    assert dataset.build_prompt(pair()) == generate.build_prompt(pair())


def test_poem_id_survives_for_later_auditing():
    assert dataset.build_example(pair())["poem_id"] == 1


# --- drop, never truncate -----------------------------------------------------

def test_overlong_pairs_are_dropped(tokenizer):
    """TRL truncates to max_length by default. A truncated poem is still scored
    against its FULL text by the grounding checker, so a quote from the removed
    region would count as grounded when the model never received it."""
    long_poem = pair(lines=["a reasonably long line of verse here"] * 400)
    assert dataset.build_dataset([long_poem], tokenizer) == []


def test_pairs_that_fit_survive(tokenizer):
    built = dataset.build_dataset([pair()], tokenizer)
    assert len(built) == 1 and built[0]["completion"]


def test_length_is_measured_on_both_halves(tokenizer):
    """Measured the way TRL builds the sequence — separately then summed — so
    the drop decision is made on the real length."""
    example = dataset.build_example(pair())
    expected = (len(tokenizer(example["prompt"])["input_ids"])
                + len(tokenizer(example["completion"])["input_ids"]))
    assert dataset.token_length(example, tokenizer) == expected


def test_the_eos_token_is_accounted_for(tokenizer):
    """TRL appends EOS, so a pair one token under the cap would exceed it."""
    import inspect
    assert "+ 1" in inspect.getsource(dataset.build_dataset)


# --- what TRL actually does with it -------------------------------------------

def collated_labels(examples, tokenizer):
    """Run examples through TRL's own preparation and return the first labels."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    trainer = SFTTrainer(
        model="gpt2",
        args=SFTConfig(output_dir="/tmp/trl_masking_check", max_steps=1,
                       report_to=[], completion_only_loss=True,
                       per_device_train_batch_size=1, eval_strategy="no",
                       max_length=config.MAX_SEQ_LEN),
        train_dataset=Dataset.from_list(examples),
        processing_class=tokenizer,
    )
    batch = next(iter(trainer.get_train_dataloader()))
    return batch["input_ids"][0], batch["labels"][0]


@pytest.mark.slow
def test_trl_masks_exactly_the_prompt(tokenizer):
    """The property everything rests on. If TRL ever moved this boundary, the
    model would train on part of its own prompt and nothing would raise."""
    from transformers import AutoTokenizer

    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    gpt2.pad_token = gpt2.eos_token
    example = dataset.build_example(pair())

    ids, labels = collated_labels([example], gpt2)
    masked = int((labels == -100).sum())
    assert masked == len(gpt2(example["prompt"])["input_ids"])


@pytest.mark.slow
def test_supervised_text_is_the_completion(tokenizer):
    from transformers import AutoTokenizer

    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    gpt2.pad_token = gpt2.eos_token
    example = dataset.build_example(pair())

    ids, labels = collated_labels([example], gpt2)
    supervised = gpt2.decode([i for i, l in zip(ids, labels) if l != -100])
    assert example["completion"] in supervised
    # The poem must NOT be supervised — that is the whole point of the masking.
    assert "first line here" not in supervised


@pytest.mark.slow
def test_completion_only_loss_changes_the_loss():
    """If the flag were doing nothing the two losses would be identical, and the
    model would be scored on predicting the poem."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    gpt2.pad_token = gpt2.eos_token
    rows = [dataset.build_example(pair())] * 4

    losses = {}
    for flag in (True, False):
        trainer = SFTTrainer(
            model=AutoModelForCausalLM.from_pretrained("gpt2"),
            args=SFTConfig(output_dir=f"/tmp/trl_loss_{flag}", max_steps=1,
                           report_to=[], completion_only_loss=flag,
                           per_device_train_batch_size=2, eval_strategy="no",
                           max_length=config.MAX_SEQ_LEN),
            train_dataset=Dataset.from_list(rows), processing_class=gpt2)
        batch = next(iter(trainer.get_train_dataloader()))
        with torch.no_grad():
            losses[flag] = trainer.model(
                **{k: v for k, v in batch.items()
                   if k in ("input_ids", "attention_mask", "labels")}).loss.item()

    assert losses[True] != losses[False]


# --- the loss is the one we think it is ---------------------------------------

def test_the_loss_is_plain_negative_log_likelihood():
    """``loss_type`` selects the OBJECTIVE, not just its implementation.

    ``chunked_nll`` is the same mathematics as ``nll`` — it only drops
    ``-100`` positions before the ``lm_head`` matmul and chunks the
    cross-entropy. ``dft`` is a different objective that reweights tokens by
    their own probability. A TRL default flipping between those two would
    change what the model optimises, and every reported perplexity with it,
    while every run still completed and every loss curve still looked normal.
    """
    from trl import SFTConfig

    resolved = SFTConfig(output_dir="/tmp/trl_loss_type_check").loss_type
    assert resolved in ("nll", "chunked_nll"), (
        f"TRL resolved loss_type to {resolved!r}. If that is 'dft' the "
        f"objective is no longer plain cross-entropy and the perplexity H4 "
        f"uses is not comparable with anything already recorded.")


def test_the_output_head_is_not_adapted():
    """``chunked_nll`` patches ``lm_head`` and TRL refuses it when a PEFT
    adapter wraps that layer. Adding ``lm_head`` to the LoRA targets would
    therefore either raise at train time or silently downgrade the loss path.
    """
    assert "lm_head" not in config.LORA_TARGET_MODULES


# --- the setting is explicit --------------------------------------------------

def test_completion_only_loss_is_set_explicitly():
    """TRL infers it from the dataset shape when unset, and an inferred
    behaviour is one a schema change could silently flip."""
    import inspect
    from src.train import loop

    assert "completion_only_loss=True" in inspect.getsource(loop.training_arguments)


# --- the alignment view -------------------------------------------------------

def test_alignment_rows_straddle_the_boundary(tokenizer):
    """The table must show the seam, or it demonstrates nothing."""
    rows = dataset.alignment_rows(dataset.build_example(pair()), tokenizer)
    assert any(r["label"] == -100 for r in rows)
    assert any(r["label"] != -100 for r in rows)


def test_alignment_marks_prompt_positions_unscored(tokenizer):
    """Every -100 row except the last one before the boundary is unscored: the
    loss is taken on what a position PREDICTS, not on the position itself."""
    rows = dataset.alignment_rows(dataset.build_example(pair()), tokenizer)
    boundary = [i for i, r in enumerate(rows) if r["label"] != -100][0]
    assert not any(r["scored"] for r in rows[:boundary - 1])
    assert all(r["scored"] for r in rows[boundary:-1])


def test_alignment_predicts_the_next_token_not_the_current(tokenizer):
    """The forward shift. If this were off by one the model would be trained to
    copy its input rather than continue it."""
    example = dataset.build_example(pair())
    rows = dataset.alignment_rows(example, tokenizer)
    scored = [r for r in rows if r["scored"]]
    for row, following in zip(scored, rows[rows.index(scored[0]) + 1:]):
        assert row["predicts"] == following["token"]


# --- perplexity ---------------------------------------------------------------

def test_perplexity_is_exp_of_the_loss():
    import math
    from src.train import loop

    assert loop.perplexity(0.0) == 1.0                 # floor, not zero
    assert abs(loop.perplexity(1.0) - math.e) < 1e-9


def test_perplexity_of_an_untrained_model_is_the_vocabulary_size():
    """The anchor that makes the loss scale readable: uniform probability over
    |V| tokens is what 'has learned nothing' looks like."""
    import math
    from transformers import AutoConfig
    from src.train import loop

    vocab = AutoConfig.from_pretrained(config.MODEL).vocab_size
    assert abs(loop.perplexity(math.log(vocab)) - vocab) < 1.0


def test_perplexity_is_clamped_so_a_diverged_run_still_records():
    import math
    from src.train import loop

    assert loop.perplexity(1e6) == math.exp(config.MAX_LOG_PERPLEXITY)


# --- gradient accumulation ----------------------------------------------------

def test_accumulation_normalises_by_token_count_not_microbatch_count():
    """``GRAD_ACCUM_STEPS=4`` only means "effective batch 16" if the four
    microbatches are weighted by how many supervised tokens each holds.

    Averaging four per-microbatch means instead weights each equally regardless
    of length — a real bug in transformers until late 2024. The fix is a
    batch-wide ``num_items_in_batch``; if a future version dropped it, every
    reported loss would shift and nothing would raise.
    """
    import inspect
    from transformers import AutoModelForCausalLM, Trainer

    assert "num_items_in_batch" in inspect.getsource(Trainer.get_batch_samples)
    forward = inspect.signature(
        AutoModelForCausalLM.from_config(
            __import__("transformers").AutoConfig.from_pretrained("gpt2")).forward)
    assert ("num_items_in_batch" in forward.parameters
            or "kwargs" in forward.parameters)
