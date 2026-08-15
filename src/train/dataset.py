"""Turn poem/interpretation pairs into the prompt-completion form TRL expects.

``SFTTrainer`` with ``completion_only_loss=True`` handles tokenisation, label
masking and EOS for a prompt-completion dataset. Verified rather than assumed:
on a worked example it masks exactly the prompt's token count and supervises
exactly the completion plus EOS.

That removes the hand-written masking this module used to carry — the
tokenise-separately-and-concatenate dance, the ``-100`` label construction, the
padding collator. All of it was correct and all of it was ours to maintain.

**Two things stay here, and both are project rules the library does not know
about.**

*Nothing is ever truncated.* TRL truncates to ``max_length`` by default. A
truncated poem would let the grounding checker match a quote against text the
model never received — scoring as grounded something the model could not have
read. Over-long pairs are dropped here, before TRL sees them, and counted.

*The prompt is the teacher's.* The student is trained to answer the question the
targets answer, so the template comes from the same place the teacher's did.
"""

from __future__ import annotations

import logging

import config

log = logging.getLogger(__name__)


def build_prompt(pair: dict) -> str:
    """Render the fixed template for one poem.

    The *same* template the teacher was given. A different prompt here would
    make every interpretation a response to something the model never saw.
    """
    return config.TEACHER_PROMPT_TEMPLATE.format(
        title=pair["title"],
        author=pair["author"],
        poem="\n".join(pair["lines"]),
        min_words=config.MIN_WORDS,
        max_words=config.MAX_WORDS,
    )


def build_example(pair: dict) -> dict:
    """One prompt-completion record.

    The poem lives in the *prompt* and the interpretation in the *completion*,
    which is what tells TRL where the loss boundary is. The model reads the
    whole sequence; only the completion is scored.
    """
    return {
        "prompt": build_prompt(pair),
        "completion": pair["interpretation"],
        "poem_id": pair["poem_id"],
    }


def token_length(example: dict, tokenizer) -> int:
    """Tokens the full sequence occupies.

    Measured on prompt and completion separately and summed, because that is
    how TRL builds the sequence — tokenising the joined string would let BPE
    merge across the seam and give a length that is off by a token or two,
    exactly at the boundary where the drop decision is made.
    """
    return (len(tokenizer(example["prompt"])["input_ids"])
            + len(tokenizer(example["completion"])["input_ids"]))


def build_dataset(pairs: list[dict], tokenizer,
                  max_length: int | None = None) -> list[dict]:
    """Prompt-completion records, dropping — never truncating — over-long pairs.

    TRL would truncate these silently. Dropping instead is a project rule with
    a measurement behind it: a truncated poem is still scored against its full
    text by the grounding checker, so a quote from the removed region would
    count as grounded when the model never received it.

    The drop count is logged rather than swallowed; it belongs in the funnel
    like any other loss.
    """
    limit = config.MAX_SEQ_LEN if max_length is None else max_length
    examples, dropped = [], 0

    for pair in pairs:
        example = build_example(pair)
        # +1 for the EOS token TRL appends.
        if token_length(example, tokenizer) + 1 <= limit:
            examples.append(example)
        else:
            dropped += 1

    if dropped:
        log.warning("%d/%d pairs exceed %d tokens and were DROPPED, not "
                    "truncated — truncation would let the grounding checker "
                    "match a quote against text the model never saw",
                    dropped, len(pairs), limit)
    log.info("built %d prompt-completion examples from %d pairs",
             len(examples), len(pairs))
    return examples


def alignment_rows(example: dict, tokenizer, context: int = 3,
                   limit: int = 6) -> list[dict]:
    """Token-by-token view across the prompt/completion boundary.

    For the notebook. Shows what the model predicts at each position and which
    of those predictions the loss is taken on, which is the one thing about the
    objective that a formula states and does not make concrete.

    Returns the last ``context`` prompt tokens and the first ``limit``
    completion tokens, so the boundary sits in the middle of the table.
    """
    prompt_ids = tokenizer(example["prompt"])["input_ids"]
    completion_ids = tokenizer(example["completion"])["input_ids"]
    ids = prompt_ids + completion_ids
    # -100 on the prompt is what TRL writes; reproduced here rather than
    # imported so the table shows the labels the trainer actually builds.
    labels = [-100] * len(prompt_ids) + completion_ids

    start = max(0, len(prompt_ids) - context)
    stop = min(len(ids), len(prompt_ids) + limit)

    rows = []
    for t in range(start, stop):
        nxt = labels[t + 1] if t + 1 < len(labels) else None
        rows.append({
            "position": t,
            "token": tokenizer.decode([ids[t]]),
            "label": labels[t],
            # The model predicts FORWARD: logits at t are scored against t+1.
            "predicts": None if nxt is None or nxt == -100
                        else tokenizer.decode([nxt]),
            "scored": nxt is not None and nxt != -100,
        })
    return rows


def describe(examples: list[dict], tokenizer) -> str:
    """A short summary for the training notebook."""
    if not examples:
        return "no examples"

    prompts = [len(tokenizer(e["prompt"])["input_ids"]) for e in examples]
    completions = [len(tokenizer(e["completion"])["input_ids"]) for e in examples]
    total = sorted(p + c for p, c in zip(prompts, completions))
    supervised = sum(completions) / sum(total)

    return (
        f"{len(examples)} examples\n"
        f"  sequence length   median {total[len(total) // 2]}  "
        f"p99 {total[int(len(total) * 0.99)]}  max {total[-1]}\n"
        f"  supervised tokens {supervised:.1%} of the total\n"
        f"  the rest is the poem and the prompt template, which the model reads "
        f"but is not scored on"
    )
