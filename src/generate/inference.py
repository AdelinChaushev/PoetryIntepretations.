"""Generate interpretations for all five arms on the evaluation poems.

**Evaluating a model on its own training data is the single easiest way to
invalidate the whole project**, and it raises nothing: the outputs look fine,
the run completes, and the result is excellent for the wrong reason.

Under 5-fold that risk lived here, as a per-poem adapter lookup — ``lora_r8``
had five adapters and each could legitimately generate only for the poems its
own fold held out. Getting the lookup wrong evaluated 120 of 150 poems on
training data. The holdout design removes the lookup entirely: every arm has
exactly one adapter, and the test poems were held out by *every* model, so the
arm name alone determines the adapter.

That moves the risk rather than abolishing it. What can still go wrong is
generating for a poem that is **not** in the test partition — a pool poem every
LoRA arm trained on. :func:`assert_routing` checks both: that each output came
from its own arm's adapter, and that its poem is one no model ever saw.

**Decoding is fixed identically across every arm** — temperature, top-p,
repetition penalty, token budget, seed. Otherwise the comparison measures
sampling settings rather than adaptation method. All five arms run in one
session, which is what makes "identical" a fact rather than an intention.
"""

from __future__ import annotations

import json
import logging

import config

log = logging.getLogger(__name__)


def build_prompt(poem: dict, arm: str, exemplars: list[dict] | None = None) -> str:
    """The prompt for one poem under one arm.

    ``base_zero`` and every LoRA arm get the **same** prompt — the one the
    teacher was given and the model was trained on. That is deliberate: it makes
    the weights the only difference between them, so ``base_zero`` vs
    ``lora_r8`` isolates the weight update and nothing else.

    ``base_few`` prepends worked examples, which is the whole of what it adds.
    Its exemplars come from a pool excluded from every evaluation fold, so a
    poem can never appear both as a prompt example and as an evaluation item.
    """
    from src.train.dataset import build_prompt as training_prompt

    body = training_prompt(poem)
    if arm != "base_few":
        return body

    assert exemplars, "base_few needs exemplars; without them it IS base_zero"
    shown = "\n\n".join(
        f"{training_prompt(example)}{example['interpretation']}"
        for example in exemplars[:config.N_FEWSHOT]
    )
    return f"{shown}\n\n{body}"


#: Arms that use no adapter at all. ``template`` needs no model either, but it
#: is generated in the same pass so every arm lands in one file written by one
#: code path.
UNTRAINED_ARMS: tuple[str, ...] = ("template", "base_zero", "base_few")


def adapter_for(arm: str) -> "object | None":
    """Which adapter generates this arm, or None if it is untrained.

    One adapter per arm under the holdout design, so this is a name lookup and
    takes no poem: the test poems were held out by every model, which is what
    makes a single global adapter per arm correct rather than a shortcut.

    Via ``config.adapter_dir``, never spelled out here: training writes the same
    path, and two spellings of one convention drift into a missing adapter for
    one arm — which reads as a partial run, not as a naming bug.
    """
    if arm in UNTRAINED_ARMS:
        return None

    rank = next((r for r in config.LORA_ARM_RANKS if arm == f"lora_r{r}"), None)
    if rank is None:
        raise ValueError(f"unknown arm {arm!r}; expected one of {config.ARMS}")
    return config.adapter_dir(rank)


def generate_one(prompt: str, model, tokenizer) -> str:
    """One interpretation, with the fixed decoding settings.

    Every parameter comes from config and none is passed per-arm, because a
    per-arm override is exactly how a comparison quietly becomes a comparison of
    sampling settings.
    """
    import torch

    # NOT truncated. Truncation here was a silent, arm-specific catastrophe:
    # `base_few` prepends three exemplars costing 1,898 tokens, so every one of
    # its prompts overran a MAX_SEQ_LEN-derived cap and was cut from the right —
    # keeping the worked examples and discarding the poem being interpreted. The
    # arm then wrote interpretations of poems it had never seen, and nothing
    # raised. An assertion is the only safe behaviour: a prompt that does not
    # fit is a bug to fix, never a prompt to shorten.
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    length = encoded["input_ids"].shape[1]
    assert length + config.GEN_MAX_NEW_TOKENS <= config.GEN_MAX_CONTEXT, (
        f"prompt is {length} tokens and generation needs "
        f"{config.GEN_MAX_NEW_TOKENS} more, over the {config.GEN_MAX_CONTEXT} "
        f"context. Truncating would silently drop the end of the prompt — for "
        f"base_few that is the target poem. Raise GEN_MAX_CONTEXT (the model "
        f"handles 32,768; this is our limit, not its) or shorten the exemplars.")

    torch.manual_seed(config.SEED)

    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=config.GEN_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=config.GEN_TEMPERATURE,
            top_p=config.GEN_TOP_P,
            repetition_penalty=config.GEN_REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
        )
    new = output[0][encoded["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


def _key(record) -> tuple:
    return (record["poem_id"], record["arm"]) if isinstance(record, dict) else record


def load_cached() -> list[dict]:
    """Arm outputs already generated, newest per (poem, arm) winning."""
    path = config.ARM_OUTPUTS_PATH
    if not path.exists():
        return []
    by_key: dict = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            by_key[_key(record)] = record
    return list(by_key.values())


def _append(record: dict) -> None:
    config.ARM_OUTPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.ARM_OUTPUTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_arm(poems: list[dict], arm: str, load_model, tokenizer=None,
                 exemplars: list[dict] | None = None,
                 test_ids: set | None = None) -> list[dict]:
    """Generate one arm for every poem, from that arm's own adapter.

    Args:
        load_model: called with an adapter path (or None) and returns a model.
            Passed in rather than imported so the caller controls loading and
            can cache the base model across arms.
        test_ids: the test partition. Checked BEFORE anything is generated,
            because the cost of finding out afterwards is a whole GPU session
            and the symptom — good scores — does not look like a failure.

    ``template`` needs no model at all but is generated in this same pass
    anyway, so every arm lands in one file written by one code path.
    """
    if test_ids is not None:
        stray = sorted({p["poem_id"] for p in poems} - set(test_ids))
        assert not stray, (
            f"{len(stray)} poem(s) are not in the test partition: {stray[:5]}. "
            f"Every LoRA arm trained on the pool, so generating for these would "
            f"evaluate a model on its own training data — which raises nothing "
            f"and reads as an excellent result.")

    done = {_key(r) for r in load_cached()}
    pending = [p for p in poems if (p["poem_id"], arm) not in done]
    log.info("%s: %d cached, %d to generate", arm, len(poems) - len(pending),
             len(pending))
    if not pending:
        return [r for r in load_cached() if r["arm"] == arm]

    # One adapter for the whole arm, loaded once.
    adapter = adapter_for(arm)
    model = None if arm == "template" else load_model(adapter)

    for poem in pending:
        text = (config.TEMPLATE_ARM_TEXT if arm == "template"
                else generate_one(build_prompt(poem, arm, exemplars),
                                  model, tokenizer))
        _append({
            "poem_id": poem["poem_id"],
            "arm": arm,
            # Recorded so the routing can be audited after the fact, not merely
            # trusted at generation time.
            "adapter": str(adapter) if adapter else None,
            "interpretation": text,
        })

    return [r for r in load_cached() if r["arm"] == arm]


def assert_routing(outputs: list[dict], test_ids: set | None = None) -> None:
    """No output was produced by a model that trained on its poem.

    Re-derived independently rather than trusting what generation recorded, and
    re-run on the laptop after the outputs come down — this is the assertion
    that catches the failure which otherwise looks like an excellent result. A
    model evaluated on its own training data scores higher on every metric and
    nothing else in the pipeline would notice.

    Two ways that can happen, and both are checked:

    1. **Wrong adapter.** An untrained arm carrying one, or ``lora_r8`` output
       generated by the ``lora_r16`` adapter. Under the holdout design this no
       longer risks training-data leakage — every adapter held out every test
       poem — but it silently relabels one arm as another, which is enough to
       make the r8-vs-r16 contrast meaningless.
    2. **Wrong poem.** A poem outside the test partition. Every LoRA arm trained
       on the pool, so this *is* the leak, and it is now the one that matters.
    """
    for record in outputs:
        arm = record["arm"]
        expected = adapter_for(arm)
        assert record["adapter"] == (str(expected) if expected else None), (
            f"poem {record['poem_id']} ({arm}) records adapter "
            f"{record['adapter']}, but {arm} generates from {expected}")

    if test_ids is not None:
        stray = sorted({r["poem_id"] for r in outputs} - set(test_ids))
        assert not stray, (
            f"{len(stray)} generated poem(s) are outside the test partition: "
            f"{stray[:5]}. Every LoRA arm trained on the pool, so these outputs "
            f"were produced by a model that had already read the poem.")

    log.info("routing verified for %d outputs across %d arm(s)%s", len(outputs),
             len({r["arm"] for r in outputs}),
             "" if test_ids is None else f" on {len(set(test_ids))} test poems")


def coverage(outputs: list[dict], poems: list[dict]) -> dict:
    """How many poems each arm produced, so a partial run is visible."""
    counts: dict = {}
    for record in outputs:
        counts[record["arm"]] = counts.get(record["arm"], 0) + 1
    return {arm: f"{counts.get(arm, 0)}/{len(poems)}" for arm in config.ARMS}


def _smoke() -> None:
    """All five arms on a handful of poems, on CPU, in under a minute.

    The routing is the assertion this exists to exercise. Evaluating a model on
    its own training data is the single easiest way to invalidate the project —
    it inflates every metric and raises nothing — so it is checked here rather
    than discovered on Kaggle.
    """
    from src.data import splits
    from src.model import setup

    config.configure_logging()
    assert config.SMOKE, "run with SMOKE=1; this is a size flag, not a mode"

    # Start clean, or a second invocation reports "0 to generate" and exercises
    # the cache instead of the generation path it exists to test. Safe because
    # SMOKE resolves this to smoke_arm_outputs.json.
    assert "smoke" in config.ARM_OUTPUTS_PATH.name, "refusing to clear real outputs"
    config.ARM_OUTPUTS_PATH.unlink(missing_ok=True)

    pairs = splits.load_training_pairs()
    holdout = splits.load_holdout()
    assert pairs and holdout["test"], "training pairs or holdout missing"

    test_ids = set(holdout["test"])
    poems = [p for p in pairs if p["poem_id"] in test_ids][:3]
    assert poems, "no test poems found in the pairs file"

    by_id = {p["poem_id"]: p for p in pairs}
    exemplars = [by_id[i] for i in holdout["exemplars"] if i in by_id]

    tokenizer = setup.load_tokenizer()
    base = setup.load_base_model()
    # One model for every arm here: the smoke run checks routing and plumbing,
    # not adapter quality, and no adapters exist before the first Kaggle run.
    for arm in UNTRAINED_ARMS:
        generate_arm(poems, arm, lambda _: base, tokenizer, exemplars,
                     test_ids=test_ids)

    outputs = [r for r in load_cached()
               if r["poem_id"] in {p["poem_id"] for p in poems}]
    assert_routing(outputs, test_ids)

    print(f"\n  coverage {coverage(outputs, poems)}")
    for arm in UNTRAINED_ARMS:
        text = next(r["interpretation"] for r in outputs if r["arm"] == arm)
        print(f"\n  {arm}: {text[:110]!r}")
    print(f"\n  routing verified; the LoRA arms are checked by "
          f"tests/test_fold_routing.py")


if __name__ == "__main__":
    _smoke()
