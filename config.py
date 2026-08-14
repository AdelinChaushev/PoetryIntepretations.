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

import logging
import os
import warnings
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

#: One row per judge from the swap test. Generated, never hand-typed, so the
#: reported numbers cannot drift from the raw scores they came from.
SWAP_SUMMARY_CSV_PATH: Path = RESULTS_DIR / "swap_test_summary.csv"

#: Checksums, git commit, package versions and the settings a result was
#: produced under. Committed with the results so a reader can verify the data
#: analysed is the data shipped, rather than having to take it on trust.
MANIFEST_PATH: Path = RESULTS_DIR / "manifest.json"
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
#: DEVIATION FROM PRE-REGISTRATION, recorded rather than quietly applied. The
#: pre-registered secondary was `gemini-2.5-flash`, which the API still lists
#: but refuses to new keys with "no longer available to new users". Substituted
#: with the nearest available pinned Flash model.
#:
#: Pinned, never `gemini-flash-latest`: an alias that moves under the project
#: would change the instrument partway through a run and make scores from
#: different days incomparable.
#:
#: This touches no headline number — the secondary judge answers exactly one
#: question, "does the conclusion flip?", and every hypothesis test is computed
#: from the primary alone.
SECONDARY_JUDGE = JudgeSpec("gemini_flash", "gemini-3.5-flash", "GOOGLE_API_KEY")

JUDGES: tuple[JudgeSpec, ...] = (PRIMARY_JUDGE, SECONDARY_JUDGE)


#: Keys may also live in a `.env` file at the project root, which `.gitignore`
#: excludes. This exists because a Jupyter server inherits its environment at
#: launch and cannot pick up a later `export` — restarting the whole server to
#: add one variable is friction that ends with someone pasting a key into a
#: tracked file. Loading `.env` at import removes that temptation entirely.
#: Real environment variables always win, so this never overrides a deliberate
#: `export`, and Kaggle (which has no `.env`) is unaffected.
ENV_FILE: Path = PROJECT_ROOT / ".env"


def _load_dotenv() -> None:
    """Load `.env` into the environment without overriding what is already set."""
    if not ENV_FILE.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional convenience only
        return
    load_dotenv(ENV_FILE, override=False)


_load_dotenv()


def require_api_key(env_var: str) -> str:
    """Return an API key from the environment, or fail with a usable message.

    Keys are never literals and never committed — they come from the shell
    environment or from a gitignored `.env`. No key should ever be pasted into
    a tracked file or a Kaggle notebook.
    """
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(
            f"{env_var} is not set.\n"
            f"Either export it in the shell that launched Jupyter:\n"
            f"    export {env_var}=...\n"
            f"or add it to {ENV_FILE} (gitignored), which needs no restart:\n"
            f"    {env_var}=..."
        )
    return key


# ---------------------------------------------------------------------------
# Data acquisition and filtering
# ---------------------------------------------------------------------------

#: PoetryDB: a fixed anthology of 129 authors, free, no API key.
POETRYDB_BASE_URL: str = "https://poetrydb.org"

#: The API errors intermittently, so retries are the expected path rather than
#: an exceptional one. Backoff doubles each attempt from this base.
FETCH_MAX_RETRIES: int = 5
FETCH_BACKOFF_SECONDS: float = 1.0
FETCH_TIMEOUT_SECONDS: float = 30.0

#: Retries for the *bulk* per-author endpoint, which has a per-title fallback.
#: Deliberately low: for the largest collections (Byron, Shelley) the 503 is
#: permanent, and exhausting FETCH_MAX_RETRIES there burns ~31s of backoff
#: before reaching a fallback that was always going to be needed. Retry hard
#: only where there is no alternative.
FETCH_BULK_RETRIES: int = 2

#: Timeout for the *bulk* per-author endpoint, shorter than the general one for
#: the same reason: that call has a fallback. PoetryDB returns its own 503 at
#: ~15s for oversized collections, so this only bites when the server hangs
#: rather than errors — and then every extra second is spent waiting for a
#: failure before running a fallback that was always going to be needed.
FETCH_BULK_TIMEOUT_SECONDS: float = 18.0

#: Retries when fetching a single poem by title, during the fallback. Also
#: deliberately low: a poem whose response repeatedly times out is a very long
#: poem, and long poems are discarded by the [MIN_LINES, MAX_LINES] filter
#: anyway. Spending 165s to retrieve something about to be thrown away is pure
#: loss, and the failure is logged and skipped rather than raised.
FETCH_TITLE_RETRIES: int = 2

#: PoetryDB returns 403 to urllib's default ``Python-urllib/x.y`` user agent,
#: so one must be set explicitly. Identifying the project is also just polite
#: to a free service.
FETCH_USER_AGENT: str = "poetry-grounding/0.1 (SoftUni course project)"

#: Authors PoetryDB serves. A fixed anthology, not a growing one, so this is a
#: property of the source rather than a target. Recorded because a partial
#: fetch is otherwise silent: the two known failures — a 403 on an unset user
#: agent and a 503 on the largest per-author collections — both end with fewer
#: authors and no error, and everything downstream would simply run on a
#: smaller corpus. Comparing against a known count makes that loud.
POETRYDB_AUTHOR_COUNT: int = 129

#: Pause between author requests. Not required by the API, but the whole
#: anthology is 129 requests against a free service — there is no reason to
#: hammer it.
FETCH_DELAY_SECONDS: float = 0.3

#: Under SMOKE, fetch only this many authors so the module runs end-to-end in
#: seconds. The full run takes all 129.
SMOKE_MAX_AUTHORS: int = 2

#: Cap on corpus size. ``None`` means NO CAP: every in-range poem gets a
#: teacher call, and the corpus is whatever survives the funnel. The surviving
#: count is measured and reported, never targeted — a target would either waste
#: teacher calls on poems that get filtered out anyway, or silently truncate a
#: corpus that came out larger than expected.
#:
#: Day-1 measurement: 3,156 poems fetched from 129 authors, of which 2,498 fall
#: within the line bounds below. What survives the rest of the funnel is not
#: knowable until the teacher has run.
N_POEMS: int | None = 20 if SMOKE else None

#: Minimum poem length, in lines — DERIVED from the measured corpus, not
#: chosen. The teacher emits 3.39 quoted spans per interpretation (the prompt
#: asks for two or three), and the share of a poem that ends up inside quotes
#: rises sharply as poems shorten:
#:
#:     8-9 lines   42% of the poem quoted (median)
#:     10-12       31%
#:     13-16       24%
#:     25-39       14%
#:     40+          7%
#:
#: Extrapolating below 8 lines, three quoted spans exceed half the poem and
#: approach the whole text. At that point the grounding check stops measuring
#: what it is for: an "interpretation" that reproduces the poem scores as
#: perfectly grounded, so quoting becomes indistinguishable from reading.
#:
#: 8 is where the measured curve crosses roughly 40%. The floor cost 239 poems
#: (2-7 lines), which is stated in limitations rather than passed over.
MIN_LINES: int = 8

#: Share of a poem's non-blank lines that must carry an ascending source line
#: number before the numbering is stripped as an artefact. A handful of
#: PoetryDB records arrive as ``"4 All skillful in the wars;"`` — the number is
#: an editorial reference, not part of the poem. Left in place it corrupts two
#: things at once: the model is trained on text containing line numbers, and a
#: correctly-quoted couplet fails the grounding check because the number sits
#: between the two lines.
#:
#: High, and paired with a monotonicity test, because the failure mode runs one
#: way. Stripping a poem that genuinely opens lines with numerals would delete
#: real words; leaving one unstripped costs a few false ungrounded verdicts.
NUMBERED_LINE_THRESHOLD: float = 0.8

#: There is deliberately NO maximum line count. An earlier version capped poems
#: at 100 lines — a number chosen by argument rather than measurement. Measuring
#: it showed that nothing under ~150 lines systematically exceeds the token
#: budget, and that line count is a poor proxy for token count anyway: a
#: 193-line poem of short lines fits, while a 120-line poem of long ones does
#: not. The cap discarded ~110 usable poems (4.4% of the corpus) by rejecting on
#: the wrong variable. MAX_POEM_TOKENS below is the constraint that genuinely
#: exists, and it is derived rather than picked.

#: Interpretation length bounds, in words. Below the floor it is too thin to
#: be an interpretation; above the ceiling it is padded.
MIN_WORDS: int = 80
MAX_WORDS: int = 250

#: Share of interpretations a single tone word may occupy before the tone slot
#: is being filled from habit rather than from reading the poem. Not a filter —
#: nothing is dropped for crossing it — but a threshold the EDA marks, because
#: this failure is invisible to every other check: an interpretation can follow
#: the schema, quote accurately, clear the funnel, and still say the same thing
#: about every poem in the corpus. Set where a word is common enough to be
#: unconditional rather than descriptive.
TONE_DOMINANCE_WARN: float = 0.5

#: Pairs sampled per condition when measuring how much author identity alone
#: predicts similarity between poems. Every same-author pair is used where
#: there are fewer than this; cross-author pairs are sampled to match, so the
#: two distributions are compared at equal n rather than 3.2M against 2.5k.
SIMILARITY_PAIRS: int = 4000

#: TF-IDF cosine above which two poems by the same author are treated as the
#: same text. PoetryDB publishes some poems under several titles — Brooke's
#: "The Soldier" appears three times, Tennyson's In Memoriam sections twice —
#: and deduplication keys on title as well as text, so these survive it.
#:
#: This threatens exactly one thing, and it is the central measurement: if the
#: "different poem by the same author" drawn for the strict swap condition is
#: actually the SAME poem under another title, that condition silently becomes
#: the matched condition, and the poem-level gap collapses to zero for a reason
#: that has nothing to do with the model. `swap_test` must exclude these.
NEAR_DUPLICATE_THRESHOLD: float = 0.9

#: Minimum poems an author needs to enter the authorship-attribution check.
#: Below this the class is too small to hold out from, and accuracy would be
#: dominated by classes with one or two examples rather than by real signal.
MIN_POEMS_FOR_ATTRIBUTION: int = 20

#: Label shuffles used for the attribution null. The reported p-value floors at
#: 1/(N+1), so 30 permutations cannot report below p≈0.032 however strong the
#: effect is — the null's *mean accuracy* is the informative output here, not
#: the p-value. Raising this costs a full refit per permutation.
N_PERMUTATIONS: int = 30

#: Share held out of the attribution check and scored exactly once. K-fold
#: already tests every poem on a model that never trained on it, but every poem
#: is still seen during some fit, so no single number rests on wholly unseen
#: data. This split does, and it is reported beside the cross-validated figure
#: as a check that the two agree rather than as a replacement for it.
HOLDOUT_FRACTION: float = 0.25

#: Hard token ceiling for prompt + target combined, measured with the real
#: tokeniser. A pair that exceeds it is DROPPED, never truncated: truncating
#: would let the grounding checker match a quote against text the model never
#: saw, silently inflating every grounding number.
MAX_SEQ_LEN: int = 512 if SMOKE else 2048

#: Tokens occupied by the prompt template plus a maximum-length interpretation,
#: measured with the real tokeniser. Kept as a constant so importing config
#: stays free of side effects; ``tests/test_filter.py`` recomputes it and fails
#: if the prompt template changes without this being updated with it.
PROMPT_OVERHEAD_TOKENS: int = 416

#: Derived, not chosen: what remains of the sequence budget once the prompt and
#: the target are accounted for. A poem longer than this cannot be trained on
#: without truncation, and truncation is never allowed — it would let the
#: grounding checker match quotes against text the model never saw.
MAX_POEM_TOKENS: int = MAX_SEQ_LEN - PROMPT_OVERHEAD_TOKENS


# ---------------------------------------------------------------------------
# Teacher generation
# ---------------------------------------------------------------------------

#: Poems in the day-1 pilot. Enough to estimate a hallucination rate with a
#: usable interval, cheap enough that a bad prompt costs cents rather than the
#: whole corpus. Generation is resumable, so the pilot is not wasted work.
PILOT_SIZE: int = 30

#: Pilot interpretations printed in full for reading. No metric answers
#: "is this any good"; that requires looking.
PILOT_SHOW: int = 2

#: Low enough that the four-part schema is followed reliably, high enough that
#: the interpretations are not all the same sentence.
#: Attempts per poem before giving up. A hallucinated quote is a bad draw from
#: the sampler, not a property of the poem — the same class of failure as a
#: timeout, which is already retried. Discarding a usable poem over a
#: recoverable output failure is the wasteful choice, and dropping is not
#: neutral either: it removes long poems (7.5% above 40 lines vs 5.0% below 12)
#: and Byron (14.7%) disproportionately, biasing the corpus toward poems the
#: teacher found easy.
#:
#: This makes each target best-of-N rather than a single sample, which is
#: ordinary rejection sampling — but it means the FIRST attempt's verdict must
#: be recorded, because the reported teacher hallucination rate is a finding
#: about the teacher, not about the corpus. Retry to save the poem, never to
#: improve the number.
GENERATE_MAX_ATTEMPTS: int = 3

#: Ceiling on attempts accumulated ACROSS runs, for `retry_ungrounded=True`.
#: `GENERATE_MAX_ATTEMPTS` is a per-run budget, so re-running would otherwise
#: give the same poem three more calls indefinitely. Some poems are genuinely
#: hard for the teacher to quote — dialect spelling, heavy elision, archaic
#: orthography — and those should stop absorbing calls rather than be retried
#: forever at the ~0.5% of the corpus where success is least likely.
GENERATE_MAX_TOTAL_ATTEMPTS: int = 9

#: Pause between teacher calls. ~2,500 requests against a paid API; a small
#: delay keeps well inside rate limits and costs a few minutes overall.
#: Abort generation if this many calls fail in a row from the start. Repeated
#: identical failures mean something systemic — expired key, billing limit,
#: outage — and grinding through 3,156 poems to report the same error 3,156
#: times wastes an hour and, if the calls are partially succeeding, real money.
GENERATE_MAX_CONSECUTIVE_FAILURES: int = 5

TEACHER_DELAY_SECONDS: float = 0.2

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

#: ``None`` is the full surviving corpus, whatever size that turns out to be.
#: The finite points straddle LIMA's 1,000-example threshold (Zhou et al. 2023)
#: so the saturation curve can be compared against theirs.
DATA_SIZE_SWEEP: tuple[int | None, ...] = (200, 500, 1000, None)
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

#: First backoff step, doubling per attempt. Rate limits are the expected
#: failure on a free tier, and a 450-call run that aborts halfway has still
#: spent the calls it made — so waiting is cheaper than restarting.
JUDGE_BACKOFF_SECONDS: float = 2.0

#: Minimum matched-minus-mismatched_random gap for a judge to be usable at all.
#: Checked on TEACHER outputs, where grounding is known, so a judge failing it
#: is failing on the easy case. Stated as an effect size rather than a p-value:
#: with 150 paired observations almost any non-zero gap is significant, and the
#: question is whether the instrument discriminates USEFULLY. One point on a
#: ten-point scale is the least that could be called separation.
MIN_JUDGE_SEPARATION: float = 1.0

#: Seconds between judge calls. Judge runs are thousands of requests against a
#: paid API; a small delay keeps well inside rate limits and costs minutes.
JUDGE_DELAY_SECONDS: float = 0.1

#: Scoring range the judge is asked for. Fixed here because the swap test is a
#: DIFFERENCE between conditions, so the scale only has to be consistent, never
#: calibrated — judge miscalibration affects matched and mismatched equally and
#: cancels. That is what lets this work without ground truth.
#: Generous, and it has to be. The reply is one integer, but current Gemini
#: Flash models reason before answering and their thinking tokens come out of
#: this budget: at 8 tokens they returned an EMPTY string, having spent the
#: whole allowance thinking. That failure is silent — an empty reply parses to
#: None and the pair is recorded as unscored, so a too-small budget would look
#: like a judge that refuses to answer rather than like a misconfiguration.
#:
#: Runaway prose is guarded by the PARSER, not by this cap. `parse_score`
#: anchors its match at the start of the reply, so a judge that writes an essay
#: yields None instead of having a stray number read as its verdict.
JUDGE_MAX_TOKENS: int = 512

JUDGE_SCORE_MIN: int = 1
JUDGE_SCORE_MAX: int = 10

#: The swap-test prompt. ONE fixed template, never changed mid-run: a mixture of
#: two prompts would make matched and mismatched scores incomparable, which is
#: the only thing the measurement depends on.
#:
#: It asks about GROUNDING, not quality. A judge asked "is this a good
#: interpretation?" would reward fluent, well-argued text regardless of which
#: poem it was written for — which is precisely the failure this project exists
#: to detect, reproduced inside its own instrument.
SWAP_JUDGE_PROMPT_TEMPLATE: str = """\
You are checking whether an interpretation was written about one specific poem.

POEM
"{title}" by {author}

{poem}

INTERPRETATION
{interpretation}

Score from {min_score} to {max_score} how specifically this interpretation \
describes THIS poem:

{min_score}-2  describes a different poem; its claims do not fit this text
3-4  generic; would fit many poems equally well
5-6  broadly consistent, but little that is specific to this poem
7-8  clearly about this poem: its imagery, argument or movement
9-{max_score}  unmistakably this poem; specific detail that fits nothing else

Do NOT judge whether the interpretation is well written, insightful or \
well argued. A fluent, elegant interpretation of the WRONG poem scores low.

Reply with a single integer from {min_score} to {max_score} and nothing else.\
"""

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

#: ``mismatched_random`` must draw from a DIFFERENT AUTHOR, or it would
#: sometimes coincide with the same-author condition and blur the two — the
#: standard and strict controls would then converge for a reason that has
#: nothing to do with the model.
MISMATCH_REQUIRES_DIFFERENT_AUTHOR: bool = True

#: Evaluation poems are sampled only from authors holding at least this many
#: poems in the corpus, so a same-author sibling always exists and all three
#: conditions are computable for every evaluation poem. Complete coverage keeps
#: the paired bootstrap clean; the cost is an evaluation set biased toward
#: prolific, heavily-anthologised poets, which must be stated in limitations —
#: and that is also where author-prior leakage is strongest, so the bias runs
#: against this project's own conclusions rather than in their favour.
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
        f"corpus cap     : {N_POEMS if N_POEMS else 'none (all survivors)'}, "
        f">= {MIN_LINES} lines, <= {MAX_POEM_TOKENS} poem tokens",
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


# ---------------------------------------------------------------------------
# Notebook logging
# ---------------------------------------------------------------------------

#: Third-party loggers that bury this project's own output.
#:
#: Each entry is silenced for a stated reason, because a blanket "quieten
#: everything" would also hide the failures worth seeing:
#:
#: ``httpx`` / ``httpcore`` / ``openai`` — one INFO line per request. Across a
#: 2,600-call teacher run that is thousands of "200 OK" lines with the real
#: failures buried inside them.
#:
#: ``huggingface_hub`` / ``filelock`` — the tokeniser is loaded from cache, but
#: the hub still logs every HEAD request it makes to revalidate it. Nothing is
#: downloaded and nothing is wrong.
#:
#: ``transformers`` — the parent, because the noise comes from several children
#: (``import_utils`` announcing that PyTorch is absent, ``utils.hub`` narrating
#: cache revalidation, ``tokenization_utils_base`` warning that a sequence
#: exceeds the model maximum). That last one is *expected and deliberate*: the
#: funnel tokenises every poem to measure it, including the 180,903-token
#: outlier, and the point of measuring is to drop what does not fit — so the
#: warning fires on exactly the poems the filter exists to catch.
_NOISY_LOGGERS = (
    "httpx", "httpcore", "openai", "urllib3",
    "huggingface_hub", "filelock",
    "transformers",
)


def configure_logging(level: int = logging.INFO) -> None:
    """Set up logging for a notebook, and quieten the libraries that shout.

    Called from the first cell of every notebook instead of ``basicConfig``, so
    the four notebooks cannot drift into four different logging setups — and so
    the reason each suppression is safe is written down once, next to the list,
    rather than rediscovered per notebook.

    ``force=True`` because Jupyter installs its own handler at kernel start; a
    plain ``basicConfig`` is a no-op once one exists, which is why log lines
    reappear after a restart when it is omitted.
    """
    logging.basicConfig(level=level, format="%(levelname)s %(message)s",
                        force=True)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    # The hub prints this via `warnings`, not logging, so a logger level does
    # not reach it — and it prints on every load. Nothing here is rate-limited:
    # the tokeniser is small and served from cache.
    warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

    # Progress bars render as control characters in a saved notebook, and the
    # notebooks are committed with outputs preserved.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


# Read when transformers and huggingface_hub are first imported, which happens
# lazily inside the funnel — long after this module loads. Set at import rather
# than inside configure_logging() so a script that never calls it is quiet too.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
