"""All figure generation.

Every figure routes through :func:`style` so that eleven-plus figures built
across several days do not drift in colour and axis convention — inconsistency
there costs visualisation marks for no reason.

Each function saves to ``config.FIGURES_DIR`` and returns the figure, so a
notebook can display it and the report can pick it up from disk.
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
    "primary": "#5c6bc0",
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
    """Write a figure to the figures directory and return it."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    log.info("wrote %s", path)
    return fig


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


def fold_sizes(folds, name: str = "02b_fold_sizes"):
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
