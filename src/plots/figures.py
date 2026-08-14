"""All figure generation.

Every figure routes through :func:`style` so that eleven-plus figures built
across several days do not drift in colour and axis convention — inconsistency
there costs visualisation marks for no reason.

Each function saves to ``config.FIGURES_DIR`` and returns the saved **path**
— see :func:`save` for why not the figure.
"""

from __future__ import annotations

import collections
import logging

import matplotlib.pyplot as plt

import config

log = logging.getLogger(__name__)

#: One palette for the whole project. Arms keep the same colour in every
#: figure, so a reader learns them once.
PALETTE = {
    "template": "#9e9e9e",
    "base_zero": "#90a4ae",
    "base_few": "#5c6bc0",
    "lora_r8": "#43a047",
    "lora_r16": "#1b5e20",
    # Generic roles for figures that are not per-arm. `primary` deliberately
    # equals `base_few` so the default colour matches the reference arm — but
    # that means a two-series plot must NOT pair them, or both series render
    # identical. Use primary/secondary, or primary/accent.
    "primary": "#5c6bc0",
    "secondary": "#00897b",
    "accent": "#e07b39",
    "reference": "#c62828",
}


def style() -> None:
    """Apply the shared matplotlib style. Called by every figure function."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


def save(fig, name: str):
    """Write a figure to the figures directory and return its path.

    Returns the **path**, not the figure, because returning the figure renders
    it twice in a notebook: the inline backend displays every open figure at
    the end of a cell, and Jupyter then renders the returned ``Figure`` again
    as the cell's last expression. A path has no rich repr, so the figure
    appears once — and the caller still gets something the report can use.
    """
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    log.info("wrote %s", path)
    return path


def corpus_statistics(corpus: list[dict], top_n: int = 12):
    """Figure 2 — line-count distribution and author coverage.

    The author panel is not decorative. Folds are grouped by author, so the
    largest collections determine how balanced the partition can be; and the
    same-author swap condition only exists for authors with more than one poem.
    The shape of this distribution constrains both.
    """
    style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    line_counts = [poem["linecount"] for poem in corpus]
    left.hist(line_counts, bins=30, color=PALETTE["primary"], edgecolor="white")
    # Only the lower bound is a line count. The upper bound is a token budget,
    # which does not correspond to any single vertical line here — line length
    # varies enough that a longer poem can fit where a shorter one does not.
    left.axvline(config.MIN_LINES, color=PALETTE["reference"],
                 linestyle="--", linewidth=1,
                 label=f"min {config.MIN_LINES} lines "
                       f"(max is {config.MAX_POEM_TOKENS} tokens)")
    left.set_xlabel("lines per poem")
    left.set_ylabel("poems")
    left.set_title("Poem length")
    left.legend()

    counts = collections.Counter(poem["author"] for poem in corpus)
    top = counts.most_common(top_n)
    names = [name for name, _ in top][::-1]
    values = [count for _, count in top][::-1]

    right.barh(names, values, color=PALETTE["primary"])
    right.set_xlabel("poems in corpus")
    right.set_title(f"Largest collections (of {len(counts)} authors)")

    singletons = sum(1 for n in counts.values() if n < 2)
    right.annotate(
        f"{singletons} authors have a single poem\n"
        f"(no same-author swap condition)",
        xy=(0.97, 0.05), xycoords="axes fraction", ha="right", fontsize=8,
        color="#555",
    )

    fig.suptitle(
        f"Corpus: {len(corpus)} poems, {len(counts)} authors — "
        f"skew constrains fold balance and the strict swap control",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    return save(fig, "02_corpus_statistics")


def fold_sizes(folds, name: str = "02g_fold_sizes"):
    """Fold balance under author grouping — a diagnostic, not a headline.

    Worth plotting because authors cannot be split, so one prolific poet can
    unbalance a fold. Seeing it beats discovering it as an odd result later.
    """
    style()
    sizes = folds.sizes()
    mean = sum(sizes) / len(sizes)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(range(len(sizes)), sizes, color=PALETTE["primary"])
    ax.axhline(mean, color=PALETTE["reference"], linestyle="--", linewidth=1,
               label=f"mean {mean:.0f}")
    ax.set_xlabel("fold")
    ax.set_ylabel("poems held out")
    ax.set_title("Fold sizes (grouped by author)")
    ax.set_xticks(range(len(sizes)))
    ax.legend()
    fig.tight_layout()
    return save(fig, name)


def interpretation_statistics(corpus: list[dict]):
    """Figure 2b — what the training targets look like.

    Three properties of the teacher's output, none of which the funnel reports
    because the funnel only records what was *rejected*.

    Length matters because the arms are compared on judge scores, and a longer
    answer can score better for reasons unrelated to grounding. Establishing
    the teacher's distribution here gives the later per-arm comparison a
    reference rather than leaving verbosity as an uncontrolled variable.

    Quote count matters because the schema asks for two or three, and the
    grounding check is only as discriminating as the number of quotes it has to
    check — an interpretation quoting once is a weaker test than one quoting
    three times.
    """
    style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    from src.eval import grounding

    words = [len(poem["interpretation"].split()) for poem in corpus]
    axes[0].hist(words, bins=30, color=PALETTE["primary"], edgecolor="white")
    for bound in (config.MIN_WORDS, config.MAX_WORDS):
        axes[0].axvline(bound, color=PALETTE["reference"], linestyle="--",
                        linewidth=1)
    axes[0].set_xlabel("words per interpretation")
    axes[0].set_ylabel("interpretations")
    axes[0].set_title("Target length")
    axes[0].annotate(f"bounds [{config.MIN_WORDS}, {config.MAX_WORDS}]\n"
                     f"median {sorted(words)[len(words) // 2]}",
                     xy=(0.97, 0.9), xycoords="axes fraction", ha="right",
                     va="top", fontsize=8, color="#555")

    quotes = [grounding.check(poem["interpretation"], poem)["n_quotes"]
              for poem in corpus]
    counts = collections.Counter(quotes)
    axes[1].bar(sorted(counts), [counts[k] for k in sorted(counts)],
                color=PALETTE["primary"])
    axes[1].set_xlabel("exact quotes per interpretation")
    axes[1].set_title("Quoting behaviour")
    axes[1].annotate(f"mean {sum(quotes) / len(quotes):.2f}",
                     xy=(0.97, 0.9), xycoords="axes fraction", ha="right",
                     va="top", fontsize=8, color="#555")

    # Share of the poem that sits inside quotes. This is the measurement the
    # 8-line floor was derived from: as poems shorten, three quoted spans
    # approach the whole text, and reproducing the poem would score as
    # perfectly grounded while saying nothing.
    share, lines = [], []
    for poem in corpus:
        quoted = sum(len(q.split())
                     for q in grounding.extract_quotes(poem["interpretation"]))
        total = len(grounding.poem_text(poem).split())
        if total:
            share.append(min(1.0, quoted / total))
            lines.append(poem["linecount"])
    axes[2].scatter(lines, share, s=6, alpha=0.25, color=PALETTE["accent"],
                    edgecolors="none")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("lines per poem (log)")
    axes[2].set_ylabel("share of poem inside quotes")
    axes[2].set_title("Why the 8-line floor exists")
    axes[2].axvline(config.MIN_LINES, color=PALETTE["reference"],
                    linestyle="--", linewidth=1,
                    label=f"floor {config.MIN_LINES} lines")
    axes[2].legend()

    fig.suptitle("The training targets: length, quoting, and the floor "
                 "the corpus is bounded by", fontsize=10, y=1.03)
    fig.tight_layout()
    return save(fig, "02b_interpretation_statistics")


def tone_vocabulary(corpus: list[dict], top_n: int = 15):
    """Figure 2c — is the tone slot filled from habit rather than from reading?

    The failure this looks for is invisible to every other check. An
    interpretation can follow the schema, quote accurately, pass the funnel,
    and still say "reflective and melancholy" for every poem in the corpus —
    carrying no information about the poem in front of it.

    That matters because these are the training targets. A word appearing in
    most outputs is a word the student would learn to emit unconditionally, and
    it would score as well-formed output while being exactly the
    interpretation-shaped text this project exists to detect.
    """
    style()
    from src.eval import format_check

    counts = format_check.part_vocabulary(
        [poem["interpretation"] for poem in corpus], part=3)
    total = len(corpus)
    top = counts.most_common(top_n)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    names = [word for word, _ in top][::-1]
    shares = [count / total for _, count in top][::-1]
    highest = max(shares) if shares else 0
    ax.barh(names, shares,
            color=[PALETTE["reference"] if s >= config.TONE_DOMINANCE_WARN
                   else PALETTE["primary"] for s in shares])
    ax.axvline(config.TONE_DOMINANCE_WARN, color=PALETTE["reference"],
               linestyle="--", linewidth=1,
               label=f"{config.TONE_DOMINANCE_WARN:.0%} of interpretations")
    ax.set_xlabel(f"share of the {total} interpretations using the word")
    ax.set_title("Tone vocabulary — the slot a template would hide in")
    ax.legend()
    fig.suptitle(
        f"Most frequent tone word appears in {highest:.1%} of interpretations",
        fontsize=10, y=1.02)
    fig.tight_layout()
    return save(fig, "02c_tone_vocabulary")


def training_pair_tokens(corpus: list[dict]):
    """Figure 2e — what the sequence budget actually has to hold.

    The number that defends ``MAX_SEQ_LEN``. Attention on the target GPUs is
    quadratic in sequence length, so the cap is a memory decision, and it is
    only defensible against the distribution it has to cover.

    Two things to read off it. The cap sits far above p99, so it does not bind
    on ordinary examples — nothing is being truncated, and the funnel's
    ``fits context`` stage dropping zero poems is the same fact stated another
    way. And the median sits far *below* the cap, which is what makes dynamic
    padding worth the trouble: padding every batch to the cap would spend most
    of the compute on padding tokens.
    """
    import numpy as np

    from src.data.filter import n_tokens, pair_text

    style()
    poem_tokens = [n_tokens("\n".join(poem["lines"])) for poem in corpus]
    pair_tokens = sorted(n_tokens(pair_text(poem, poem["interpretation"]))
                         for poem in corpus)
    median = pair_tokens[len(pair_tokens) // 2]
    p99 = pair_tokens[int(len(pair_tokens) * 0.99)]
    poem_median = sorted(poem_tokens)[len(poem_tokens) // 2]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4))

    # Distinct colours and mismatched styles: PALETTE["primary"] and
    # PALETTE["base_few"] are the same value, so a two-series plot must not
    # pair them or both render identically.
    bins = np.linspace(0, config.MAX_SEQ_LEN, 45)
    left.hist(pair_tokens, bins=bins, color=PALETTE["primary"], alpha=0.85,
              label="poem + interpretation (what is trained on)")
    left.hist(poem_tokens, bins=bins, histtype="step", linewidth=1.8,
              color=PALETTE["accent"], label="poem alone")
    left.axvline(config.MAX_SEQ_LEN, color=PALETTE["reference"], linestyle="--",
                 linewidth=1.4, label=f"MAX_SEQ_LEN = {config.MAX_SEQ_LEN}")
    left.axvline(median, color=PALETTE["primary"], linestyle=":", linewidth=1.4,
                 label=f"median pair = {median}")
    left.set_xlim(0, config.MAX_SEQ_LEN * 1.02)
    left.set_xlabel("tokens")
    left.set_ylabel("examples")
    left.set_title("Training sequence length")
    left.legend(fontsize=8, loc="upper right")
    left.annotate(f"the offset between the two is the prompt\n"
                  f"scaffolding plus the interpretation:\n"
                  f"~{median - poem_median} tokens added to every poem",
                  xy=(0.97, 0.45), xycoords="axes fraction", ha="right",
                  fontsize=8, color="#555")

    # Longest first. Read left to right this answers "how many examples need at
    # least this much of the budget?", and the shaded region above the curve is
    # the padding that a fixed-cap batch would spend on nothing.
    waste = 1 - (sum(pair_tokens) / (len(pair_tokens) * config.MAX_SEQ_LEN))
    shares = [t / config.MAX_SEQ_LEN for t in reversed(pair_tokens)]
    right.plot(range(len(shares)), shares, color=PALETTE["primary"],
               linewidth=1.4)
    right.fill_between(range(len(shares)), shares, 1.0,
                       color=PALETTE["reference"], alpha=0.08)
    right.axhline(1.0, color=PALETTE["reference"], linestyle="--", linewidth=1.2,
                  label="the cap")
    # Anchored at zero: starting at the data minimum would make a distribution
    # that mostly sits near 30% of the cap look like it climbs to meet it.
    right.set_ylim(0, 1.03)
    right.set_xlabel("examples, longest first")
    right.set_ylabel("share of MAX_SEQ_LEN used")
    right.set_title("How much of the budget each example needs")
    right.legend(fontsize=8, loc="lower left")
    right.annotate(
        f"median {median} tokens ({median / config.MAX_SEQ_LEN:.0%} of cap)\n"
        f"p99 {p99}   max {max(pair_tokens)}\n"
        f"shaded area = padding waste if every batch\n"
        f"were padded to the cap ({waste:.0%})",
        xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
        fontsize=8, color="#555")

    fig.suptitle("The cap does not bind, and the median is far below it — "
                 "which is why padding is dynamic", fontsize=10, y=1.02)
    fig.tight_layout()
    return save(fig, "02f_training_pair_tokens")


def quote_positions(corpus: list[dict], bins: int = 20):
    """Figure 2f — where in the poem the teacher quotes from.

    A teacher that reads only the opening would still pass every grounding
    check, because a quote from line 2 is as verbatim as a quote from line 40.
    Position is the thing that separates reading a poem from skimming it, and
    nothing else measured here would notice the difference.

    It also bears on the length bias. If quotes cluster near the start, then the
    tail of a long poem is never examined, and the corpus's bias toward shorter
    work matters less than the raw drop counts suggest — the teacher was
    effectively treating long poems as short ones anyway.
    """
    style()
    from src.eval import grounding

    positions, by_length = [], []
    for poem in corpus:
        body = grounding.poem_text(poem)
        if not body:
            continue
        for quote in grounding.extract_quotes(poem["interpretation"]):
            found = body.find(grounding.normalise(quote))
            if found >= 0:
                relative = found / len(body)
                positions.append(relative)
                by_length.append((poem["linecount"], relative))

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4))

    left.hist(positions, bins=bins, color=PALETTE["primary"], edgecolor="white")
    left.axhline(len(positions) / bins, color=PALETTE["reference"],
                 linestyle="--", linewidth=1.2, label="uniform (no positional bias)")
    left.set_xlabel("position in poem (0 = first line, 1 = last)")
    left.set_ylabel("quotes")
    left.set_title("Where quotes come from")
    left.legend(fontsize=8)

    # Does the bias worsen as poems get longer? If the teacher reads a fixed
    # prefix rather than the whole text, the mean position should fall.
    buckets = [(8, 12), (13, 20), (21, 40), (41, 80), (81, 10**6)]
    labels, means = [], []
    for low, high in buckets:
        sample = [rel for lines, rel in by_length if low <= lines <= high]
        if sample:
            labels.append(f"{low}-{high if high < 10**6 else '+'}")
            means.append(sum(sample) / len(sample))
    right.bar(labels, means, color=PALETTE["primary"])
    right.axhline(0.5, color=PALETTE["reference"], linestyle="--", linewidth=1.2,
                  label="0.5 = quotes centred on the poem")
    right.set_ylim(0, 1)
    right.set_xlabel("poem length (lines)")
    right.set_ylabel("mean quote position")
    right.set_title("Does the teacher read to the end of long poems?")
    right.legend(fontsize=8)

    fig.suptitle(f"Quote position across {len(positions)} located quotes — "
                 f"a check no grounding rate would fail", fontsize=10, y=1.02)
    fig.tight_layout()
    return save(fig, "02d_quote_positions")


def author_signal(corpus: list[dict], seed: int | None = None):
    """Figure 2g — how much a poem reveals about who wrote it.

    This underwrites the strict swap condition. That condition scores an
    interpretation against a different poem *by the same author*, because an
    author's themes and manner recur: a model that learned "Dickinson writes
    about death and immortality" could emit a plausible interpretation for an
    unseen Dickinson poem without reading it, and that text would still beat a
    random Whitman poem. If author identity carried no signal, the strict
    condition would be no stricter than the standard one and decomposing the
    gap into an author-level component would be measuring nothing.

    Two panels, because cosine similarity alone is too blunt to settle it.

    **Left — pairwise similarity.** Intuition only: are same-author poems more
    alike than cross-author ones? It reads shared vocabulary, so it conflates
    what a poem is *about* with who wrote it.

    **Right — authorship attribution.** The real measurement. A linear model
    predicts the author from held-out text, scored against the majority-class
    baseline. Run over three feature sets, because *which* of them carries the
    signal changes what it means:

    ``content words``
        topic and vocabulary. High accuracy here could just mean poets write
        about different subjects.
    ``function words``
        the, of, and, but — words with no subject matter at all. This is
        classical stylometry: accuracy here is **style**, independent of topic,
        and it is the harder result to explain away.
    ``character n-grams``
        morphology, spelling and diction together; usually the strongest, and
        the least interpretable.

    Function-word accuracy well above baseline means author identity is
    recoverable from *manner* alone — which is precisely the leak the
    same-author condition exists to close.
    """
    import random

    from sklearn.feature_extraction.text import (ENGLISH_STOP_WORDS,
                                                 TfidfVectorizer)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                         permutation_test_score,
                                         train_test_split)
    from sklearn.pipeline import make_pipeline

    style()
    from src.data.filter import near_duplicate_ids
    from src.eval import grounding

    rng = random.Random(config.SEED if seed is None else seed)
    texts = [grounding.poem_text(poem) for poem in corpus]

    # PoetryDB publishes some poems under several titles, so a "same author"
    # pair can be the same text twice. Those score ~1.0 and would inflate the
    # very number this figure establishes — evidence for its own conclusion.
    duplicates = near_duplicate_ids(corpus)

    by_author: dict[str, list[int]] = collections.defaultdict(list)
    for index, poem in enumerate(corpus):
        by_author[poem["author"]].append(index)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    # --- panel 1: pairwise similarity ---------------------------------------
    matrix = TfidfVectorizer(min_df=2, stop_words="english").fit_transform(texts)
    groups = [g for g in by_author.values() if len(g) > 1]

    same = []
    while len(same) < config.SIMILARITY_PAIRS:
        group = rng.choice(groups)
        a, b = rng.choice(group), rng.choice(group)
        if a != b and corpus[b]["poem_id"] not in duplicates.get(
                corpus[a]["poem_id"], ()):
            same.append((a, b))

    authors = list(by_author)
    cross = []
    while len(cross) < len(same):
        first, second = rng.choice(authors), rng.choice(authors)
        if first != second:
            cross.append((rng.choice(by_author[first]),
                          rng.choice(by_author[second])))

    def scores(pairs):
        return [float(cosine_similarity(matrix[a], matrix[b])[0][0])
                for a, b in pairs]

    same_scores, cross_scores = scores(same), scores(cross)
    same_mean = sum(same_scores) / len(same_scores)
    cross_mean = sum(cross_scores) / len(cross_scores)

    left.hist([cross_scores, same_scores], bins=40, density=True, range=(0, 0.3),
              label=[f"different author ({cross_mean:.3f})",
                     f"same author ({same_mean:.3f})"],
              color=[PALETTE["base_few"], PALETTE["accent"]], alpha=0.8)
    left.set_xlabel("TF-IDF cosine similarity")
    left.set_ylabel("density")
    # AUC, not the ratio of means. Both means sit near zero on a heavily
    # right-skewed distribution, where a ratio exaggerates separation; AUC is
    # the probability a random same-author pair outscores a random cross-author
    # one, and 0.5 is the no-signal point.
    picks = rng.sample(range(len(cross_scores)), 2000)
    auc = sum((same_scores[i] > cross_scores[j])
              + 0.5 * (same_scores[i] == cross_scores[j])
              for i, j in zip(rng.sample(range(len(same_scores)), 2000),
                              picks)) / 2000
    left.set_title(f"Lexical similarity — AUC {auc:.3f}")
    left.legend(fontsize=8)
    left.annotate(f"AUC {auc:.3f} (0.5 = no signal)\n"
                  f"ratio of means {same_mean / cross_mean:.2f}x, but both means\n"
                  f"are near zero and skewed — a weak effect\n"
                  f"shared words only: conflates topic with authorship",
                  xy=(0.97, 0.55), xycoords="axes fraction", ha="right",
                  fontsize=8, color="#555")

    # --- panel 2: authorship attribution ------------------------------------
    keep = [(index, poem["author"]) for author, indices in by_author.items()
            if len(indices) >= config.MIN_POEMS_FOR_ATTRIBUTION
            for index in indices for poem in [corpus[index]]]
    sample = [texts[i] for i, _ in keep]
    labels = [a for _, a in keep]
    n_authors = len(set(labels))
    baseline = max(collections.Counter(labels).values()) / len(labels)

    feature_sets = {
        "content\nwords": TfidfVectorizer(min_df=2, stop_words="english"),
        "function\nwords": TfidfVectorizer(vocabulary=sorted(ENGLISH_STOP_WORDS)),
        "character\n3-5 grams": TfidfVectorizer(analyzer="char_wb",
                                                ngram_range=(3, 5), min_df=3),
    }
    # Shuffled, seeded folds so the number is reproducible rather than an
    # artefact of the corpus's author-sorted order.
    splitter = StratifiedKFold(4, shuffle=True, random_state=config.SEED)
    model = LogisticRegression(max_iter=2000, C=10)

    # Cross-validation uses every poem as test exactly once, but every poem is
    # also seen during some fit. This split is touched once, at the end, so the
    # headline number rests on poems no fit has ever seen.
    train_texts, holdout_texts, train_labels, holdout_labels = train_test_split(
        sample, labels, test_size=config.HOLDOUT_FRACTION, stratify=labels,
        random_state=config.SEED)

    names, accuracies, spreads = [], [], []
    null_accuracy, holdout_accuracy = None, None
    for name, vectoriser in feature_sets.items():
        # The vectoriser is fitted INSIDE each fold, not once over everything.
        # Fitting it first would compute idf weights — and, for the content-word
        # and character sets, the vocabulary itself — from the held-out poems,
        # which inflates accuracy by leaking the test fold into the features.
        pipeline = make_pipeline(vectoriser, model)
        folds = cross_val_score(pipeline, sample, labels, cv=splitter, n_jobs=-1)
        names.append(name)
        accuracies.append(folds.mean())
        spreads.append(folds.std())
        if "function" in name:
            # The honest null. A majority-class baseline says what a constant
            # predictor scores; this says what the SAME pipeline scores when
            # the labels carry no information, which is what rules out leakage.
            _, permuted, _ = permutation_test_score(
                pipeline, sample, labels, cv=splitter,
                n_permutations=config.N_PERMUTATIONS, n_jobs=-1,
                random_state=config.SEED)
            null_accuracy = permuted.mean()

            holdout = make_pipeline(vectoriser, model).fit(train_texts,
                                                           train_labels)
            holdout_accuracy = holdout.score(holdout_texts, holdout_labels)

    # Fold spread is reported in each bar's label rather than drawn as error
    # bars: at this scale the caps are visual noise on top of a 40-point gap,
    # and the number is more precise than a whisker anyone has to eyeball.
    right.bar(names, accuracies,
              color=[PALETTE["base_few"], PALETTE["accent"], PALETTE["lora_r8"]])
    right.axhline(baseline, color=PALETTE["reference"], linestyle="--",
                  linewidth=1.2,
                  label=f"majority class ({baseline:.1%})")
    if null_accuracy is not None:
        right.axhline(null_accuracy, color="#555", linestyle=":", linewidth=1.2,
                      label=f"labels shuffled ({null_accuracy:.1%}, "
                            f"{config.N_PERMUTATIONS} perms)")
    right.set_ylim(0, 1)
    right.set_ylabel("4-fold accuracy")
    right.set_title(f"Predicting the author ({n_authors} authors, "
                    f"{len(sample)} poems)")
    right.legend(fontsize=8, loc="upper right")
    for index, (accuracy, spread) in enumerate(zip(accuracies, spreads)):
        right.annotate(f"{accuracy:.1%}\n±{spread:.1%}", xy=(index, accuracy),
                       ha="center", va="bottom", fontsize=8.5)
    if holdout_accuracy is not None:
        right.annotate(
            f"function words on a single untouched\n"
            f"{config.HOLDOUT_FRACTION:.0%} holdout ({len(holdout_labels)} poems): "
            f"{holdout_accuracy:.1%}",
            xy=(0.02, 0.97), xycoords="axes fraction", ha="left", va="top",
            fontsize=8, color="#555")

    fig.suptitle("Weak lexical signal, strong stylometric one: function words "
                 "alone identify the author far above the shuffled-label null",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    return save(fig, "02e_author_signal")
