"""Single source of truth for the whole project.

Every hyperparameter, model name, path, seed and threshold lives here. Nothing
numeric belongs in a function body elsewhere in the codebase — if a module
needs a number, it imports it from this file.

Two switches change what the values mean:

``SMOKE``
    Size flag, not an environment flag. Shrinks everything so a script runs
    end-to-end on a CPU laptop in under a minute, to surface bugs before a
    Kaggle session is spent on them. It does **not** mean "running locally":
    the data and evaluation stages run locally at full size.

``IS_KAGGLE``
    Detected, never set by hand. Only training and generation run on Kaggle;
    everything else runs on the laptop. This switch resolves paths, and nothing
    else.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------

SMOKE: bool = os.getenv("SMOKE", "0") == "1"

#: True inside a Kaggle notebook session. Kaggle sets KAGGLE_KERNEL_RUN_TYPE;
#: the directory check is a fallback for cases where it is absent.
IS_KAGGLE: bool = bool(os.getenv("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle").is_dir()

#: One seed for everything that can be seeded: fold assignment, evaluation
#: sampling, exemplar choice, weight init, generation. Fixing it in one place
#: is what makes the fold assignment reproducible across machines.
SEED: int = 42


# ---------------------------------------------------------------------------
# Paths
#
# Never hardcode /kaggle/input or ./data at a call site. Resolve through these.
# Both can be overridden by environment variable, which is how a Kaggle
# notebook points at a specific input dataset without a code change.
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent


def _resolve_dir(env_var: str, kaggle_default: str, local_default: Path) -> Path:
    """Resolve a directory from environment, then Kaggle, then local layout."""
    override = os.getenv(env_var)
    if override:
        return Path(override)
    return Path(kaggle_default) if IS_KAGGLE else local_default


#: Corpus, cache and fold assignment. Read-only on Kaggle.
DATA_DIR: Path = _resolve_dir("POETRY_DATA_DIR", "/kaggle/input", PROJECT_ROOT / "data")

#: Everything the project produces. Kaggle wipes this between sessions, which
#: is why adapters and generated outputs must be saved as a Kaggle Dataset
#: before a session ends.
RESULTS_DIR: Path = _resolve_dir(
    "POETRY_RESULTS_DIR", "/kaggle/working", PROJECT_ROOT / "results"
)

RAW_POEMS_PATH: Path = DATA_DIR / "raw_poems.jsonl"
INTERPRETATIONS_PATH: Path = DATA_DIR / "interpretations.jsonl"
CORPUS_PATH: Path = DATA_DIR / "corpus.jsonl"
FOLD_ASSIGNMENT_PATH: Path = DATA_DIR / "folds.json"
EVAL_POEMS_PATH: Path = DATA_DIR / "eval_poems.jsonl"

RUNS_CSV_PATH: Path = RESULTS_DIR / "runs.csv"
PRIOR_WORK_CSV_PATH: Path = RESULTS_DIR / "prior_work_comparison.csv"
ARM_OUTPUTS_PATH: Path = RESULTS_DIR / "arm_outputs.json"
ADAPTERS_DIR: Path = RESULTS_DIR / "adapters"
FIGURES_DIR: Path = RESULTS_DIR / "figures"


def ensure_dirs() -> None:
    """Create the writable output directories.

    Called explicitly rather than at import, so importing config stays free of
    side effects and remains safe inside tests.
    """
    for directory in (RESULTS_DIR, ADAPTERS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    if not IS_KAGGLE:
        DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Models
#
# Teacher and judge are deliberately from different families. An evaluator
# scoring its own family's generations favours them (Panickssery et al. 2024).
# ---------------------------------------------------------------------------

#: Student. Base, not Instruct — an instruction-tuned checkpoint would already
#: follow the output schema, which would contaminate the format-compliance
#: measurement. ``gpt2`` under SMOKE is a tiny stand-in for fast CPU runs, not
#: a fallback model: the project is committed to Qwen with no fallback.
MODEL: str = "gpt2" if SMOKE else "Qwen/Qwen2.5-0.5B"

TEACHER_MODEL: str = "deepseek-chat"
TEACHER_BASE_URL: str = "https://api.deepseek.com"
TEACHER_API_KEY_ENV: str = "DEEPSEEK_API_KEY"

class JudgeSpec(NamedTuple):
    """One judge: which model it is, and where its credentials come from."""

    name: str
    model: str
    api_key_env: str
    base_url: str | None = None


#: Pre-registered PRIMARY judge. Every hypothesis test (H1-H4) and every headline
#: number is computed from this judge alone.
PRIMARY_JUDGE = JudgeSpec("gpt4o_mini", "gpt-4o-mini", "OPENAI_API_KEY")

#: SECONDARY judge, run as a pre-registered robustness check — never averaged
#: with the primary. Averaging would produce a composite nobody can interpret,
#: and choosing which judge to report after seeing results is exactly the
#: freedom the pre-registration exists to remove. The secondary answers one
#: question: does the conclusion flip?
#:
#: A different family from the teacher (DeepSeek), the student (Qwen) and the
#: primary judge (OpenAI), so agreement between the two is evidence about the
#: measurement rather than about shared lineage.
SECONDARY_JUDGE = JudgeSpec("gemini_flash", "gemini-2.5-flash", "GOOGLE_API_KEY")

JUDGES: tuple[JudgeSpec, ...] = (PRIMARY_JUDGE, SECONDARY_JUDGE)


def require_api_key(env_var: str) -> str:
    """Return an API key from the environment, or fail with a usable message.

    Keys are never literals and never committed. They are only ever needed
    locally — no key should be pasted into a Kaggle notebook.
    """
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(
            f"{env_var} is not set. Export it in your shell before running:\n"
            f"    export {env_var}=..."
        )
    return key


# ---------------------------------------------------------------------------
# Data acquisition and filtering
# ---------------------------------------------------------------------------

POETRYDB_BASE_URL: str = "https://poetrydb.org"
FETCH_MAX_RETRIES: int = 5
FETCH_BACKOFF_SECONDS: float = 1.0

#: Target corpus size after the full filtering funnel. PoetryDB is a fixed
#: anthology, so verify on day 1 that this many poems survive before relying
#: on it — report the real number if it falls short.
N_POEMS: int = 20 if SMOKE else 2000

#: Line-count bounds. A cheap pre-filter only; MAX_SEQ_LEN below is the real
#: constraint. The upper bound is set by GPU memory and step time across ~19
#: training runs, not by the model's context window — Qwen2.5-0.5B has 32K.
MIN_LINES: int = 8
MAX_LINES: int = 100

#: Interpretation length bounds, in words. Below the floor it is too thin to
#: be an interpretation; above the ceiling it is padded.
MIN_WORDS: int = 80
MAX_WORDS: int = 250

#: Hard token ceiling for prompt + target combined, measured with the real
#: tokeniser. A pair that exceeds it is DROPPED, never truncated: truncating
#: would let the grounding checker match a quote against text the model never
#: saw, silently inflating every grounding number.
MAX_SEQ_LEN: int = 512 if SMOKE else 2048


# ---------------------------------------------------------------------------
# Teacher generation
# ---------------------------------------------------------------------------

#: Low enough that the four-part schema is followed reliably, high enough that
#: the interpretations are not all the same sentence.
TEACHER_TEMPERATURE: float = 0.4
TEACHER_MAX_TOKENS: int = 512

#: ONE fixed template. Changing it partway through generation would make the
#: corpus a mixture of two tasks and confound every later comparison. Notebooks
#: display this constant rather than retyping the prompt, so documentation
#: cannot drift from what actually ran.
#:
#: Part 2 is the load-bearing one: it forces a checkable claim about the source
#: text, which is what makes grounding measurable without a reference answer.
TEACHER_PROMPT_TEMPLATE: str = """\
You are writing a short interpretation of a poem.

Title: {title}
Author: {author}

{poem}

Write an interpretation in exactly these four labelled parts:

1. Central idea - one or two sentences on what the poem is about.
2. Key images - two or three images from the poem. For each one, quote the
   exact line, copied word for word from the poem above.
3. Tone - the tone of the poem, in a few words.
4. Interpretive claim - one specific claim about what the poem is doing, of a
   kind a reader could disagree with.

Rules:
- Quote lines exactly as they appear above. Never paraphrase inside quotation
  marks.
- Write between {min_words} and {max_words} words in total.
- Do not add any section beyond the four listed.
"""


# ---------------------------------------------------------------------------
# Splitting: 5-fold grouped cross-validation, grouped by author
# ---------------------------------------------------------------------------

N_FOLDS: int = 5

#: Folds are GROUPED BY ACT, not cut at the section level. Sections of one Act
#: are not independent samples — they share defined terms, drafting style,
#: subject matter and cross-references — so training on one section of an Act
#: carries information about every other section of it. Splitting at the
#: section level would put correlated items on both sides of the partition,
#: the same violation as letting one patient's scans straddle train and test.
#: This is grouped cross-validation (cf. sklearn's GroupKFold).
#:
#: No Act ever appears in both a training partition and its own evaluation set.
#: The cost is uneven fold sizes, since Acts vary enormously in length.
FOLD_GROUP_KEY: str = "author"

#: Largest tolerated deviation from equal fold size, as a fraction. Grouping by
#: Act means folds cannot be balanced exactly; one very long Act can unbalance
#: the partition badly enough to matter. Exceeding this should raise on day 1,
#: while the corpus can still be rebalanced, rather than being discovered as a
#: strange result on day 7.
MAX_FOLD_SIZE_IMBALANCE: float = 0.25

#: Evaluation poems sampled from each fold's held-out portion. Sampling rather
#: than judging every held-out poem is what keeps the judge budget fixed
#: regardless of corpus size: 5 x 30 = 150 judged poems whether the corpus is
#: 800 or 2000.
EVAL_PER_FOLD: int = 2 if SMOKE else 30

#: Total evaluation set size, derived — never write 150 anywhere else.
N_EVAL_POEMS: int = N_FOLDS * EVAL_PER_FOLD

#: In-context examples for the ``base_few`` arm. Reserved from a pool that is
#: excluded from every fold's evaluation set, so a poem can never be both a
#: prompt exemplar and an evaluation item. Chosen deterministically from
#: (corpus, SEED) rather than pinned by hand, so the choice is reproducible.
N_FEWSHOT: int = 3


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

LORA_TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj")
LORA_RANK: int = 8
LORA_ALPHA: int = 16
LORA_DROPOUT: float = 0.05

LEARNING_RATE: float = 2e-4
WEIGHT_DECAY: float = 0.01
BATCH_SIZE: int = 1 if SMOKE else 8
GRAD_ACCUM_STEPS: int = 1 if SMOKE else 2

#: Fixed STEP budget, not epochs. With a 2000-poem corpus, fixed epochs would
#: make every run 2.5x longer and push the sweep past the 30 GPU-hour weekly
#: budget. It also removes a confound: at fixed epochs the 2000-poem sweep
#: point would get 2.5x the gradient updates of the 200-poem point, so the
#: data-size axis would measure compute as much as data.
MAX_STEPS: int = 5 if SMOKE else 1000
WARMUP_STEPS: int = 2 if SMOKE else 100
LR_SCHEDULE: str = "cosine"

#: Label id for positions excluded from the loss. Loss is masked to
#: interpretation tokens only: prompt positions and padding are both -100.
IGNORE_INDEX: int = -100

CHECKPOINT_EVERY_STEPS: int = 250

#: Sweep axes, varied ONE AT A TIME with the others at the defaults above.
#: Not a full grid. DATA_SIZE_SWEEP deliberately straddles LIMA's 1,000-example
#: threshold so the curve can be compared against Zhou et al. 2023.
RANK_SWEEP: tuple[int, ...] = (4, 8, 16)
LR_SWEEP: tuple[float, ...] = (1e-4, 2e-4, 5e-4)
DATA_SIZE_SWEEP: tuple[int, ...] = (200, 500, 1000, 2000)
MASKING_SWEEP: tuple[str, ...] = ("masked", "unmasked")

#: Adapter ranks used by the final evaluated arms.
ARM_RANKS: dict[str, int] = {"lora_r8": 8, "lora_r16": 16}


# ---------------------------------------------------------------------------
# Generation
#
# Fixed identically across ALL five arms. If decoding differed between arms the
# comparison would measure sampling settings rather than adaptation method.
# ---------------------------------------------------------------------------

ARMS: tuple[str, ...] = ("template", "base_zero", "base_few", "lora_r8", "lora_r16")

GEN_TEMPERATURE: float = 0.7
GEN_TOP_P: float = 0.9
GEN_REPETITION_PENALTY: float = 1.2
GEN_MAX_NEW_TOKENS: int = 400

#: The ``template`` arm: one fixed generic interpretation emitted for every
#: poem. The trivial baseline. It may score surprisingly well, which is a
#: legitimate result rather than a bug — whatever it scores is how much of a
#: "good" score is available without reading anything.
TEMPLATE_ARM_TEXT: str = """\
1. Central idea: The poem reflects on the passage of time and the way memory
   shapes how experience is understood.
2. Key images: The poem uses natural imagery to suggest change, and contrasts
   light and darkness to mark shifts in feeling.
3. Tone: Reflective and somewhat melancholy.
4. Interpretive claim: The poem suggests that understanding arrives only after
   the moment it would have been useful has passed.
"""


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

#: Every judged pair is run twice with the answer order swapped, so position
#: bias can be measured rather than assumed. Compare the resulting flip rate
#: against Zheng et al. 2023 (Table 2: GPT-4 judge consistency 65.0%).
JUDGE_BOTH_ORDERINGS: bool = True
JUDGE_TEMPERATURE: float = 0.0
JUDGE_MAX_RETRIES: int = 3

#: Both judges score everything: the swap test, the pairwise win rates, and the
#: day-2 validation on teacher outputs. Roughly 6,900 calls in total. Judge
#: outputs are keyed by judge name so the two are never silently pooled.
JUDGE_ALL_ARMS_BOTH_JUDGES: bool = True

#: The swap test scores one interpretation against three different poems.
#:
#: ``matched``
#:     the poem the interpretation was written for
#: ``mismatched_random``
#:     a random poem by a DIFFERENT author — the standard control
#: ``mismatched_same_author``
#:     a different poem by the SAME author — the strict control
#:
#: The third condition exists because an author's themes recur across their
#: poems. A model that learned "Dickinson writes about death and nature" can
#: emit a plausible interpretation for an unseen Dickinson poem without reading
#: it, and that text still beats a random Whitman poem — so the standard
#: control alone would score it as well grounded.
SWAP_CONDITIONS: tuple[str, ...] = (
    "matched",
    "mismatched_random",
    "mismatched_same_author",
)

#: ``mismatched_random`` must draw from a DIFFERENT Act, or it would sometimes
#: coincide with the same-Act condition and blur the two.
MISMATCH_REQUIRES_DIFFERENT_AUTHOR: bool = True

#: Evaluation sections are sampled only from Acts with at least this many
#: sections in the corpus, so a same-Act sibling always exists and all three
#: conditions are computable for every evaluation section. Complete coverage
#: keeps the paired bootstrap clean; the cost is an evaluation set biased
#: toward longer Acts, which must be stated in limitations.
MIN_POEMS_PER_AUTHOR_FOR_EVAL: int = 2

# --- Contamination probe -----------------------------------------------------
# Public-domain poetry is public domain because it is old and widely
# reproduced, which is exactly what puts it in web-scale pretraining data. The
# base model has very likely seen these poems already. That cannot be avoided
# by choosing different poems — contemporary poetry outside pretraining is
# under copyright and cannot be sent to an API — so it is measured instead.

#: Lines of a poem given to the base model before asking it to continue. If it
#: reproduces the rest verbatim, it memorised the poem rather than reading it.
CONTAMINATION_PROMPT_LINES: int = 2

#: Fraction of the remaining lines that must be reproduced for a poem to count
#: as memorised. Not 1.0: near-verbatim recall with small variations is still
#: recall.
CONTAMINATION_MATCH_THRESHOLD: float = 0.8

#: The probe uses GREEDY decoding, unlike generation for the arms. Sampling
#: would let a memorised poem fail the probe by chance, understating
#: contamination. This is the one place decoding deliberately differs.
CONTAMINATION_TEMPERATURE: float = 0.0

BOOTSTRAP_ITERATIONS: int = 100 if SMOKE else 10_000
CONFIDENCE_LEVEL: float = 0.95
ALPHA: float = 0.05


# ---------------------------------------------------------------------------

def summary() -> str:
    """Human-readable dump of the resolved configuration.

    Printed at the top of every notebook so the settings a result was produced
    under are recorded alongside the result.
    """
    lines = [
        f"SMOKE          : {SMOKE}",
        f"IS_KAGGLE      : {IS_KAGGLE}",
        f"seed           : {SEED}",
        "",
        f"student model  : {MODEL}",
        f"teacher model  : {TEACHER_MODEL}",
        f"judge (primary): {PRIMARY_JUDGE.model}",
        f"judge (2nd)    : {SECONDARY_JUDGE.model}  [robustness only]",
        "",
        f"corpus target  : {N_POEMS} poems, {MIN_LINES}-{MAX_LINES} lines",
        f"interpretation : {MIN_WORDS}-{MAX_WORDS} words",
        f"max seq len    : {MAX_SEQ_LEN} tokens (drop, never truncate)",
        "",
        f"folds          : {N_FOLDS}, grouped by {FOLD_GROUP_KEY}",
        f"eval poems     : {EVAL_PER_FOLD} per fold = {N_EVAL_POEMS} total",
        "",
        f"lora rank      : {LORA_RANK} (alpha {LORA_ALPHA})",
        f"target modules : {', '.join(LORA_TARGET_MODULES)}",
        f"max steps      : {MAX_STEPS} (fixed steps, not epochs)",
        f"batch size     : {BATCH_SIZE} x {GRAD_ACCUM_STEPS} accum",
        "",
        f"data dir       : {DATA_DIR}",
        f"results dir    : {RESULTS_DIR}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
