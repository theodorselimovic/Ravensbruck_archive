"""NER word cloud visualisation.

Generates one word cloud per entity type from the NER extraction output,
plus a combined 2x2 panel figure.

Usage:
    python 06_ner_wordclouds.py
    python 06_ner_wordclouds.py --min-count 3 --verbose
    python 06_ner_wordclouds.py --exclude "Ger,Pol,App,Rev"
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "ner" / "entities.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "ner" / "figures"

ENTITY_COLOURS = {
    "PER": "#e63946",
    "LOC": "#457b9d",
    "ORG": "#2a9d8f",
    "DATE": "#e9c46a",
}

ENTITY_LABELS = {
    "PER": "Persons",
    "LOC": "Locations",
    "ORG": "Organisations",
    "DATE": "Dates",
}

# Noise entities that appear across multiple types due to model confusion
DEFAULT_EXCLUDE = {"Ger", "Ger.", "Pol", "App", "Rev", "of", "the"}

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================================
# Data Loading
# ============================================================================

def load_entity_frequencies(
    input_path: Path,
    min_count: int = 2,
    exclude: Optional[Set[str]] = None,
) -> Dict[str, Counter]:
    """Load entities.csv and return frequency counters per entity type."""
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} entity rows")

    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    if exclude:
        before = len(df)
        df = df[~df["text"].isin(exclude)]
        logger.info(f"Excluded {before - len(df):,} noise entities")

    freqs: Dict[str, Counter] = {}
    for etype in sorted(df["entity_type"].unique()):
        subset = df[df["entity_type"] == etype]
        counts = Counter(subset["text"].values)
        counts = Counter({k: v for k, v in counts.items() if v >= min_count})
        freqs[etype] = counts
        logger.info(f"  {etype}: {len(counts):,} unique entities (min_count={min_count})")

    return freqs


# ============================================================================
# Word Cloud Generation
# ============================================================================

def make_wordcloud(
    frequencies: Counter,
    colour: str,
    max_words: int = 150,
    width: int = 1200,
    height: int = 600,
) -> WordCloud:
    """Generate a single word cloud from frequency counts."""

    def colour_func(*args, **kwargs):
        return colour

    wc = WordCloud(
        width=width,
        height=height,
        max_words=max_words,
        background_color="white",
        color_func=colour_func,
        prefer_horizontal=0.7,
        min_font_size=8,
        max_font_size=120,
        relative_scaling=0.5,
    )
    wc.generate_from_frequencies(frequencies)
    return wc


def save_individual_clouds(
    freqs: Dict[str, Counter],
    output_dir: Path,
    max_words: int = 150,
) -> List[Path]:
    """Save one word cloud PNG per entity type."""
    paths = []
    for etype, counts in freqs.items():
        if not counts:
            logger.warning(f"  Skipping {etype}: no entities")
            continue

        colour = ENTITY_COLOURS.get(etype, "#333333")
        label = ENTITY_LABELS.get(etype, etype)
        wc = make_wordcloud(counts, colour, max_words=max_words)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{label} ({etype})", fontsize=18, fontweight="bold", pad=12)
        ax.axis("off")

        path = output_dir / f"wordcloud_{etype.lower()}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(path)
        logger.info(f"  Saved {path.name}")

    return paths


def save_combined_panel(
    freqs: Dict[str, Counter],
    output_dir: Path,
    max_words: int = 150,
) -> Optional[Path]:
    """Save a 2x2 panel with all four entity types."""
    types_with_data = [t for t in ["PER", "LOC", "ORG", "DATE"] if freqs.get(t)]
    if len(types_with_data) < 2:
        logger.warning("Not enough entity types for a combined panel")
        return None

    ncols = 2
    nrows = (len(types_with_data) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 7 * nrows))
    axes = axes.flatten()

    for i, etype in enumerate(types_with_data):
        colour = ENTITY_COLOURS.get(etype, "#333333")
        label = ENTITY_LABELS.get(etype, etype)
        wc = make_wordcloud(freqs[etype], colour, max_words=max_words, width=800, height=400)

        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"{label} ({etype})", fontsize=16, fontweight="bold", pad=8)
        axes[i].axis("off")

    for j in range(len(types_with_data), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Named Entities in Ravensbrück Testimonies",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = output_dir / "wordcloud_combined.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved {path.name}")
    return path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate word clouds from NER extraction results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=DEFAULT_INPUT,
        help=f"Input entities CSV (default: {DEFAULT_INPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output directory for figures (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--min-count", type=int, default=2,
        help="Minimum entity frequency to include (default: 2)",
    )
    parser.add_argument(
        "--max-words", type=int, default=150,
        help="Maximum words per cloud (default: 150)",
    )
    parser.add_argument(
        "--exclude", type=str, default=None,
        help="Comma-separated entities to exclude (adds to defaults)",
    )
    parser.add_argument(
        "--no-default-exclude", action="store_true",
        help="Don't apply default noise exclusions",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # ---- Banner ----
    logger.info("=" * 60)
    logger.info("NER WORD CLOUDS")
    logger.info("=" * 60)
    logger.info(f"Input:     {args.input}")
    logger.info(f"Output:    {args.output}")
    logger.info(f"Min count: {args.min_count}")

    # ---- Exclusions ----
    exclude = set() if args.no_default_exclude else set(DEFAULT_EXCLUDE)
    if args.exclude:
        exclude.update(e.strip() for e in args.exclude.split(","))
    if exclude:
        logger.info(f"Excluding: {sorted(exclude)}")

    # ---- Load ----
    freqs = load_entity_frequencies(args.input, min_count=args.min_count, exclude=exclude)

    # ---- Generate ----
    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 60)
    logger.info("GENERATING WORD CLOUDS")
    logger.info("=" * 60)

    save_individual_clouds(freqs, args.output, max_words=args.max_words)
    save_combined_panel(freqs, args.output, max_words=args.max_words)

    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
