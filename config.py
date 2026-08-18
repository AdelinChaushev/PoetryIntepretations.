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

def _detect_kaggle() -> bool:
    """True only inside a real Kaggle session.

    The environment variable is the reliable signal. The directory fallback
    exists for sessions that do not set it, but it must name the *writable
    working directory* rather than ``/kaggle`` alone — a bare ``/kaggle`` can
    exist elsewhere, and on Colab it does. That misdetection sent DATA_DIR to
    ``/kaggle/input`` and RESULTS_DIR to ``/kaggle/working`` on a machine where
    neither exists, so the corpus read as empty and the sweep reported zero
    recorded runs while the files sat in the checkout.

    Both paths must be present: Kaggle always mounts them together.
    """
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    return Path("/kaggle/working").is_dir() and Path("/kaggle/input").is_dir()


#: Detected, never set by hand. Only training and generation run on Kaggle.
IS_KAGGLE: bool = _detect_kaggle()

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
#: The funnel's output: each surviving poem joined with its teacher
#: interpretation, so one record is one training example. Named for its contents
#: rather than "corpus", which describes `raw_poems.jsonl` — the poems alone —
#: and gives no hint that the interpretations are inside.
#:
#: Written locally and shipped to Kaggle. NOT rebuilt there: the funnel is a
#: pure function of the raw files and the thresholds, so rebuilding works right
#: up until a threshold changes, at which point the GPU side would train on a
#: different set of poems than the fold assignment describes — and nothing
#: would raise, because the ids still resolve.
TRAINING_PAIRS_PATH: Path = DATA_DIR / "training_pairs.jsonl"

#: Fold assignment, the evaluation poem ids with their folds, and the exemplar
#: ids — all three in one file, because they are one decision and separating
#: them invites shipping a stale half.
FOLD_ASSIGNMENT_PATH: Path = DATA_DIR / "folds.json"

#: The OUTER split that replaces the fold scheme: one author-disjoint test set
#: fixed before any hyperparameter is chosen, plus the pool everything else
#: trains on.
#:
#: A separate file from folds.json rather than an overwrite. The fold-based
#: runs, their adapters and the first contamination probe all key on the old
#: evaluation ids, and the report shows both designs — why 5-fold was chosen,
#: what it cost, and why a single holdout replaced it. Overwriting would make
#: that comparison unreproducible.
HOLDOUT_PATH: Path = DATA_DIR / "holdout.json"

#: Poems in the test set. 150 keeps the judge budget identical to the fold
#: design (150 x 5 arms x 3 conditions x 2 judges), so the two are comparable.
#: The realised size is slightly larger because whole authors are taken.
TEST_SIZE: int = 150

#: Folds used for hyperparameter tuning ONLY — never to define held-out data.
#: Three rather than five: tuning costs k runs per configuration, and five
#: would be 26 GPU-hours before the final model exists.
TUNING_FOLDS: int = 3

#: Poems the tuning folds are built over. Capped because tuning cost scales
#: with it, and hyperparameter *rankings* are more stable across data scale
#: than absolute losses are. A real approximation, and it belongs in the
#: writeup rather than a footnote.
TUNING_SUBSAMPLE: int = 1000

def _result(name: str) -> Path:
    """A results path that smoke runs cannot contaminate.

    Under ``SMOKE`` the file is prefixed, so a five-step gpt2 run never lands in
    a file real results are read from. This is not tidiness — each of these
    files has a route by which a smoke row would do damage silently:

    * ``runs.csv`` — ``sweep.select_winner`` minimises validation loss, and a
      tiny model on six examples can post a lower one than a real run, which
      would return a winning configuration that was never actually trained.
    * ``arm_outputs.json`` — ``inference.load_cached`` keys on
      ``(poem_id, arm)`` and takes the newest, so a smoke record silently
      **replaces** a real generation rather than sitting beside it.
    * ``contamination.jsonl`` — ``probe_all`` skips ids already present, so a
      smoke record would suppress the real probe for that poem entirely.
    """
    return RESULTS_DIR / (f"smoke_{name}" if SMOKE else name)


RUNS_CSV_PATH: Path = _result("runs.csv")
PRIOR_WORK_CSV_PATH: Path = RESULTS_DIR / "prior_work_comparison.csv"

#: One row per judge from the swap test. Generated, never hand-typed, so the
#: reported numbers cannot drift from the raw scores they came from.
SWAP_SUMMARY_CSV_PATH: Path = RESULTS_DIR / "swap_test_summary.csv"

#: Checksums, git commit, package versions and the settings a result was
#: produced under. Committed with the results so a reader can verify the data
#: analysed is the data shipped, rather than having to take it on trust.
MANIFEST_PATH: Path = RESULTS_DIR / "manifest.json"
ARM_OUTPUTS_PATH: Path = _result("arm_outputs.json")
ADAPTERS_DIR: Path = RESULTS_DIR / "adapters"


def run_adapter_dir(run_name: str) -> Path:
    """Where a run's adapter is written, keyed on the RUN NAME.

    Every run keeps its weights, which is both tidier and more defensible: each
    row in ``runs.csv`` is then traceable to the model that produced it, and a
    sweep configuration that later turns out to matter can be inspected rather
    than retrained.

    Keyed on the run name because ``(rank, fold)`` is not unique across a sweep —
    three learning rates at rank 8 all resolve to the same ``lora_r8_fold0`` and
    would overwrite each other, leaving one file belonging to no recorded run in
    particular. The run name encodes everything that varied, which is exactly the
    property needed here.

    :func:`adapter_dir` is the same path for final runs, because ``final_specs``
    names them ``lora_r{rank}_fold{fold}``. The two cannot drift: one delegates
    to the other.
    """
    return ADAPTERS_DIR / f"{'smoke_' if SMOKE else ''}{run_name}"


def adapter_dir(rank: int) -> Path:
    """Canonical location of one FINAL adapter — the ones generation loads.

    **One definition, because two would drift silently.** Training writes the
    adapter and `generate.inference.adapter_for` reads it back, and if those
    spellings ever disagree the failure is not a clean "file not found" for
    every poem — it is a missing adapter for *one* arm, which looks like a
    partial run rather than a naming bug.

    The rank is in the name because ``lora_r8`` and ``lora_r16`` are separate
    arms trained from the same code path. There is **no fold in the name**: the
    holdout design trains one adapter per arm on the whole pool, so the name of
    the arm and the name of the run are the same string, and routing needs no
    lookup at all. Under 5-fold this took a ``fold`` argument and the lookup was
    the single easiest way to invalidate the project.

    Smoke runs get their own prefix, so a five-step gpt2 adapter never occupies
    the path a real one is loaded from. Both training and generation resolve
    through here, so the two stay consistent in either mode.
    """
    return run_adapter_dir(f"lora_r{rank}")
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
    #: Reasoning budget, sent as ``reasoning_effort`` when set and omitted
    #: otherwise. ``None`` means the provider default and is required for models
    #: that reject the parameter — GPT-4o-mini is not a reasoning model and
    #: errors if it is sent.
    #:
    #: ``"none"`` matters for cost and for reliability, not for speed. A
    #: reasoning judge spends hundreds of invisible output tokens deliberating
    #: before it emits one digit, and those are billed; it also overruns its
    #: token budget mid-thought on the hardest pairs, which is where every
    #: unparseable reply in this project came from.
    reasoning: str | None = None


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
SECONDARY_JUDGE = JudgeSpec("gemini_flash", "gemini-3.5-flash", "GOOGLE_API_KEY",
                            reasoning="none")

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
#: gpt2's context is 1,024 tokens, and SMOKE uses all of it. An earlier 512
#: left only 96 tokens for a poem after the prompt template, so EVERY real poem
#: was dropped and the smoke run trained on an empty dataset — which is exactly
#: the class of bug SMOKE exists to catch, found by running it.
MAX_SEQ_LEN: int = 1024 if SMOKE else 2048

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

#: Every linear layer, attention AND MLP. The MLP is ~88% of each layer and is
#: where associative knowledge generally sits, so adapting only attention would
#: risk confirming H2/H3 by construction — the hypotheses predict that weight
#: updates do not improve grounding, and never touching the part of the network
#: where grounding would live is not a fair test of that.
#:
#: Costs 4.4M trainable parameters (0.89% of the model) against 737k for
#: attention alone. Still an 8 MB adapter that commits to git.
#:
#: gpt2 under SMOKE has a different architecture — one fused ``c_attn`` instead
#: of separate projections — so the names must differ or peft matches nothing
#: and silently trains zero parameters.
LORA_TARGET_MODULES: tuple[str, ...] = ("c_attn",) if SMOKE else (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
LORA_RANK: int = 8

#: LoRA scales its update by ``alpha / r``. Fixing alpha while sweeping rank
#: would vary the scaling 4x -> 2x -> 1x across {4, 8, 16}, so the rank axis
#: would measure capacity AND effective step size together — and the Hu et al.
#: saturation comparison would be uninterpretable. Deriving alpha from r keeps
#: the scaling constant at 2, leaving rank as the only thing that changes.
LORA_ALPHA_MULTIPLIER: int = 2


def lora_alpha(rank: int = LORA_RANK) -> int:
    """Alpha for a given rank, holding the alpha/r scaling constant."""
    return LORA_ALPHA_MULTIPLIER * rank


LORA_ALPHA: int = lora_alpha(LORA_RANK)
LORA_DROPOUT: float = 0.05

LEARNING_RATE: float = 2e-4
WEIGHT_DECAY: float = 0.01
#: 4 x 4 rather than 8 x 2 — identical effective batch, half the peak
#: activation memory. Attention is quadratic in sequence length, and it is the
#: LONGEST batch that runs out of memory, not the median: the p99 training pair
#: is ~1,800 tokens, so batches near the cap do occur. At the 2048 cap the
#: attention matrix is ~0.88 GB per layer at batch 8 and ~0.44 GB at batch 4.
#: Gradient accumulation makes the update identical either way, so the smaller
#: micro-batch buys headroom for nothing.
BATCH_SIZE: int = 1 if SMOKE else 4
GRAD_ACCUM_STEPS: int = 1 if SMOKE else 4

#: The budget: an epoch ceiling, converted to steps per run, clamped, and cut
#: short by early stopping. See MAX_EPOCHS below for why neither a fixed step
#: count nor a fixed epoch count works alone.
#: Budget expressed in EPOCHS, converted to steps per run because the step
#: count depends on how many examples that run trains on. A ceiling, not a
#: target: early stopping ends most runs sooner.
#:
#: Nine passes is where the full corpus sat under the previous fixed-step
#: budget, kept because nothing observed yet argues against it — and the
#: validation curve now decides when a run actually stops.
MAX_EPOCHS: int = 1 if SMOKE else 9

#: Floor on optimisation steps, which matters for the small data-size points.
#: Nine epochs of 200 poems is ~113 steps and 10% of that is a warmup barely
#: long enough to reach the target learning rate; a run that short would be
#: undertrained rather than data-limited, and the data-size curve would measure
#: the wrong thing.
MIN_TRAIN_STEPS: int = 2 if SMOKE else 200

#: Hard ceiling regardless of dataset size, so one run cannot consume a session.
MAX_STEPS: int = 5 if SMOKE else 1000


def steps_for(n_examples: int, epochs: int | None = None) -> int:
    """Optimisation steps covering ``epochs`` passes, clamped to the bounds.

    Expressed here rather than at the call site so every run — sweep, fold,
    smoke — derives its budget the same way, and so the epoch count is a
    property of the configuration rather than of whoever launched the run.
    """
    import math

    per_epoch = max(1, n_examples // (BATCH_SIZE * GRAD_ACCUM_STEPS))
    raw = math.ceil((MAX_EPOCHS if epochs is None else epochs) * per_epoch)
    return max(MIN_TRAIN_STEPS, min(MAX_STEPS, raw))

#: Clamp on the loss before exponentiating it into a perplexity. `exp` diverges
#: fast, so a diverged run would otherwise write an unusable number into
#: runs.csv or overflow outright. 20 is far above anything meaningful: uniform
#: probability over Qwen's 151,936-token vocabulary — a model that has learned
#: nothing at all — gives log(151936) = 11.93.
MAX_LOG_PERPLEXITY: float = 20.0

#: Warmup as a FRACTION of the run, not a fixed count. 100 steps is 10% of a
#: 1000-step run but 30% of a 330-step one, so a constant silently changes
#: meaning whenever the budget moves.
WARMUP_RATIO: float = 0.10
WARMUP_STEPS: int = max(1, int(MAX_STEPS * WARMUP_RATIO))

#: Early stopping is ON everywhere. With an epoch-based ceiling the question
#: "did this configuration overfit?" is answered per run by its own validation
#: curve, which is both the honest budget for a hyperparameter search — each
#: config compared at ITS best, not at an arbitrary shared step count — and the
#: only thing that keeps the small data-size points from training to 80 epochs.
#:
#: The consequence is that runs no longer share a step count. That is the right
#: trade: comparing under-trained against over-trained at equal compute is not
#: a fairer comparison, only a more uniform one.
#: Hard ceiling on the author-grouped validation slice. Authors are held back in
#: whole blocks, so aiming at 10% can overshoot when one author holds a large
#: share — on a small fold that produced more validation than training, which is
#: silent: the run completes, having trained on almost nothing.
VALIDATION_MAX_FRACTION: float = 0.20

EARLY_STOPPING: bool = True
EARLY_STOPPING_PATIENCE: int = 3

#: How much better an evaluation must be to count as progress.
#:
#: EarlyStoppingCallback defaults this to 0.0, which means an improvement of
#: 0.0001 resets the patience counter — so a run whose validation loss creeps
#: never accumulates three strikes and burns the entire step budget. Observed:
#: the unmasked sweep run went 2.383 -> 2.377 -> 2.387 and kept training, while
#: the n=1000 run, improving in real increments, stopped itself at epoch 4.5 of
#: 8.9.
#:
#: 0.001 is below the smallest difference that would change a conclusion here —
#: reported perplexities differ by tenths — and above the noise that was keeping
#: runs alive.
EARLY_STOPPING_THRESHOLD: float = 0.001

#: Restore the weights from the best validation step rather than the last.
#: Without this, early stopping SAVES the overfitted model it stopped because
#: of — the run ends `patience` evaluations past its own best.
RESTORE_BEST_WEIGHTS: bool = True
EVAL_EVERY_STEPS: int = 1 if SMOKE else 50
LR_SCHEDULE: str = "cosine"

#: Label id for positions excluded from the loss. Loss is masked to
#: interpretation tokens only: prompt positions and padding are both -100.
IGNORE_INDEX: int = -100

CHECKPOINT_EVERY_STEPS: int = 250

#: Sweep axes, varied ONE AT A TIME with the others at the defaults above.
#: Not a full grid: rank and learning rate are swept as a product because they
#: interact, while data size and masking are single-axis because neither is a
#: hyperparameter being tuned.
RANK_SWEEP: tuple[int, ...] = (4, 8, 16)
#: NARROWED from (1e-4, 2e-4, 5e-4) on 2026-08-16, for two reasons stated
#: together because neither alone would justify it.
#:
#: Compute: a full-corpus run costs ~1.6 GPU-hours, a session was lost to a
#: 12-hour cap, and the six final runs — which everything downstream needs —
#: had not started. The third rate was the affordable thing to drop.
#:
#: Evidence: the two rates measured differ by 2x and produced validation losses
#: identical to the seventh decimal (1.6148772 vs 1.6148783), with gradient
#: clipping already active at the lower one. The objective is flat across this
#: range, so a third point was unlikely to separate.
#:
#: **The cost is real and belongs in the limitations**: the highest rate was
#: never tried, so the selected rate is the best of two rather than of three,
#: and a better rate above the range explored cannot be ruled out.
LR_SWEEP: tuple[float, ...] = (1e-4, 2e-4)

#: How much training data changes what fine-tuning achieves. ``None`` is the
#: full surviving corpus, whatever size that turns out to be.
#:
#: This axis is not hyperparameter tuning — the final runs use all the data
#: regardless. It exists because "does more data improve GROUNDING, or only
#: format?" is one of the fine-tuning capability questions the project is about,
#: and the answer is a curve rather than a number. Format compliance and the
#: grounding gap can flatten at different points, and that divergence is the
#: finding.
#:
#: The points happen to straddle LIMA's 1,000-example threshold, which gives
#: Zhou et al. 2023 as a reference line when reading where the curve flattens.
#: That is a comparison the design pays for anyway, not the reason the axis
#: exists.
DATA_SIZE_SWEEP: tuple[int | None, ...] = (200, 500, 1000, None)
MASKING_SWEEP: tuple[str, ...] = ("masked", "unmasked")

#: How the sweep picks a winner. Fixed HERE, before any sweep run, because
#: choosing the criterion after seeing results is the researcher freedom the
#: pre-registration exists to remove — and with 13 runs there is always a metric
#: under which some config looks best.
#:
#: Validation loss, lowest wins. Ties broken by FEWER trainable parameters, so a
#: config that matches another's loss with less capacity is preferred — which is
#: also the direction Hu et al.'s saturation finding predicts.
#:
#: Note what this is NOT: judge scores never select a config. They are the
#: outcome measure, and selecting on them would fit the hyperparameters to the
#: thing being reported.
SWEEP_SELECTION_METRIC: str = "final_val_loss"
SWEEP_SELECTION_LOWER_IS_BETTER: bool = True

# ---------------------------------------------------------------------------
# Generation
#
# Fixed identically across ALL five arms. If decoding differed between arms the
# comparison would measure sampling settings rather than adaptation method.
# ---------------------------------------------------------------------------

ARMS: tuple[str, ...] = ("template", "base_zero", "base_few", "lora_r8", "lora_r16")

#: The ranks the two LoRA arms are trained at. **Pre-registered, and NOT taken
#: from whatever the sweep chose.** They name the arms; letting a winning rank
#: of 4 rename them would change the experiment after seeing results, which is
#: the researcher freedom the pre-registration exists to remove. The sweep
#: supplies the learning rate, and the rank question is answered by the CV curve
#: over {4, 8, 16} rather than by renaming an arm.
#:
#: Both are trained on the SAME pool under the holdout design, so lora_r8 vs
#: lora_r16 differs in rank and nothing else. Under 5-fold it compared fold-0 r8
#: against fold-0 r16 and the difference also carried which poems each had seen.
LORA_ARM_RANKS: tuple[int, ...] = (8, 16)

#: The headline LoRA arm — the one H1-H3 are stated about and the one the
#: data-size and masking ablations are drawn at, so the ablations' full-corpus
#: point IS this arm's own row rather than a sixth run nobody reports.
PRIMARY_LORA_RANK: int = 8

assert tuple(f"lora_r{r}" for r in LORA_ARM_RANKS) == ARMS[-len(LORA_ARM_RANKS):], (
    "ARMS and LORA_ARM_RANKS disagree; generation routes on the arm name and "
    "training writes the adapter from the rank, so a mismatch means one arm "
    "generates from the base model with nothing raising")
assert PRIMARY_LORA_RANK in LORA_ARM_RANKS

GEN_TEMPERATURE: float = 0.7
GEN_TOP_P: float = 0.9
GEN_REPETITION_PENALTY: float = 1.2
GEN_MAX_NEW_TOKENS: int = 400

#: Context available at GENERATION, and deliberately larger than MAX_SEQ_LEN.
#:
#: MAX_SEQ_LEN = 2048 is a *training* limit. It exists because attention is
#: quadratic and, at an effective batch of 16 with activations kept for the
#: backward pass, a T4 cannot hold more. None of that applies here: generation
#: runs at batch 1 under no_grad, so the only per-token cost is the KV cache,
#: which is a few tens of MB at these lengths. Qwen2.5-0.5B itself handles
#: 32,768 tokens.
#:
#: **Sized from the arm that needs it, and the reason is a bug this replaces.**
#: `base_few` prepends three worked examples, costing 1,898 tokens before the
#: target poem appears; its prompts run to a median of 2,245 tokens and a
#: maximum of 3,641. Capped at MAX_SEQ_LEN - GEN_MAX_NEW_TOKENS = 1,648, all 150
#: were right-truncated -- keeping the exemplars and cutting off the poem being
#: interpreted. `base_few` would have written interpretations of poems it never
#: saw, scored near zero on grounding, and made `base_few` vs `lora_r8` -- the
#: headline contrast -- an artifact of truncation.
#:
#: 6144 leaves ~50% headroom over the 4,041 the longest case needs. Identical
#: for every arm, because a per-arm context limit would be a per-arm handicap.
GEN_MAX_CONTEXT: int = 6144

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

#: Tokens generated per probe. Sized against CONTAMINATION_SCORED_LINES, not
#: against whole poems: emitting the scored window costs 79 tokens at the 95th
#: percentile and 134 at the worst case, so this leaves roughly 2-3x headroom
#: for a model that drifts before recovering the text.
CONTAMINATION_MAX_NEW_TOKENS: int = 256

#: Lines shorter than this are ignored when scoring recall. "And" or "O!" turn
#: up in almost any continuation by chance, and counting them would push every
#: poem's score toward the memorisation threshold.
CONTAMINATION_MIN_LINE_WORDS: int = 3

#: How many of a poem's remaining lines recall is scored over. A FIXED window,
#: not the whole poem, and the reason is a measurement bug this replaces.
#:
#: Scoring the whole remainder against a fixed token budget made the threshold
#: mechanically unreachable for long poems: at 256 tokens, 26.8% of the corpus
#: could not have scored 0.8 however perfectly the model recited — 100% of
#: poems under 20 lines were reachable against 0.3% of those over 60. The
#: `memorised` flag would then have meant "memorised AND short", and since that
#: flag stratifies every headline result, the non-memorised stratum would have
#: been quietly full of long poems the probe never gave room to.
#:
#: A fixed window asks every poem the same question — "recite the next six
#: lines" — so the flag means one thing corpus-wide. Six because the corpus
#: floor is 8 lines and the prompt takes 2, so only 0.5% of poems have fewer.
CONTAMINATION_SCORED_LINES: int = 6

#: Per-poem probe results. Committed: the memorised flag stratifies every
#: headline result, so a reader checking the stratification needs the flags,
#: not just the summary.
CONTAMINATION_PATH: Path = _result("contamination.jsonl")

BOOTSTRAP_ITERATIONS: int = 100 if SMOKE else 10_000
CONFIDENCE_LEVEL: float = 0.95
ALPHA: float = 0.05

#: Confidence level for every interval. Stated once so a figure and a table can
#: never disagree about what their error bars mean.
CI_LEVEL: float = 0.95

#: Bootstrap resamples. 10,000 is the usual floor for a stable percentile
#: interval at the 95% level — below it the endpoints wobble between runs, which
#: would make a reported CI depend on the seed.
BOOTSTRAP_RESAMPLES: int = 10_000

#: Whether to correct the four pre-registered p-values for multiple comparisons.
#:
#: OFF, deliberately. H1-H4 are not a family being screened for any significant
#: result; each is a separate pre-registered prediction with its own direction,
#: and two of them (H2, H3) predict nulls — where correction makes a null easier
#: to obtain and so favours the prediction. Holm-corrected values are reported
#: alongside as a robustness check, never substituted.
CORRECT_MULTIPLE_COMPARISONS: bool = False
MULTIPLE_COMPARISON_METHOD: str = "holm"


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
