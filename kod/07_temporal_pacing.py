"""Narrative pacing analysis of Ravensbrück testimonies.

Classifies DATE entities by their implied temporal grain (day, week,
month, year, habitual, calendar, season) and analyses how pacing
changes through each testimony. Produces per-document pacing profiles,
corpus-level statistics, and visualisations.

Usage:
    python 07_temporal_pacing.py
    python 07_temporal_pacing.py --verbose
    python 07_temporal_pacing.py --sections 5 --verbose
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENTITIES = PROJECT_ROOT / "data" / "ner" / "entities.csv"
DEFAULT_SENTENCES = PROJECT_ROOT / "data" / "ner" / "entities_by_sentence.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "ner" / "temporal"

PACING_CATEGORIES = ["day", "week", "month", "year", "habitual", "calendar", "season", "noise"]

CATEGORY_COLOURS = {
    "day": "#e63946",
    "week": "#f4a261",
    "month": "#2a9d8f",
    "year": "#264653",
    "habitual": "#e9c46a",
    "calendar": "#457b9d",
    "season": "#8ecae6",
    "noise": "#cccccc",
}

CATEGORY_LABELS = {
    "day": "Day-level",
    "week": "Week-level",
    "month": "Month-level",
    "year": "Year-level",
    "habitual": "Habitual",
    "calendar": "Calendar date",
    "season": "Seasonal",
    "noise": "Noise",
}

# Word-to-number mapping for duration extraction
WORD_NUMBERS = {
    "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "half": 0.5, "couple": 2, "few": 3, "several": 4,
}

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================================
# Temporal Expression Classification
# ============================================================================

# Pre-compiled regex patterns, tested in priority order

# Noise: age expressions, prisoner numbers, fractions, OCR artifacts
_RE_AGE = re.compile(
    r"^\d+-year-old$|^\d+ years? old$|^\w+-year-old$", re.IGNORECASE
)
_RE_PRISONER_NUM = re.compile(r"^\d{4,}$")
_RE_FRACTION = re.compile(r"^one-quarter$|^one-half$|^one-third$", re.IGNORECASE)
_RE_OCR_NOISE = re.compile(r"^[\d\s\+\[\]/\\]+$")

# Habitual: recurring temporal patterns
_RE_HABITUAL = re.compile(
    r"(?:^daily$|^weekly$|^monthly$|^yearly$"
    r"|^every\b|^each\b"
    r"|^all\s+day$|^all-day$"
    r"|^the\s+(?:entire|whole|working|full)\s+day$"
    r"|^an\s+entire\s+day$"
    r"|^Sundays?$|^Mondays?$|^Tuesdays?$|^Wednesdays?$"
    r"|^Thursdays?$|^Fridays?$|^Saturdays?$"
    r")",
    re.IGNORECASE,
)

# Calendar: specific dates (year, month-year, day-month-year)
_RE_CALENDAR_FULL = re.compile(
    r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December)\s+\d{4}",
    re.IGNORECASE,
)
_RE_CALENDAR_MONTH_YEAR = re.compile(
    r"(?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December)\s+\d{4}",
    re.IGNORECASE,
)
_RE_CALENDAR_YEAR = re.compile(r"^(?:19\d{2})(?:[–\-]\d{1,2})?$")
_RE_CALENDAR_RANGE = re.compile(
    r"(?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December|\d{1,2})\s.*\bto\b",
    re.IGNORECASE,
)

# Season patterns
_SEASON_WORDS = {"spring", "summer", "autumn", "fall", "winter", "christmas", "easter"}
_RE_SEASON = re.compile(
    r"(?:^(?:the\s+)?(?:spring|summer|autumn|fall|winter)"
    r"|^(?:the\s+)?(?:spring|summer|autumn|fall|winter)\s+(?:of\s+)?\d{4}"
    r"|^Christmas|^Easter"
    r"|^the\s+(?:autumn|winter|spring|summer)\s+(?:of|and)"
    r"|^(?:early|late|mid-?)\s*(?:spring|summer|autumn|fall|winter)"
    r")",
    re.IGNORECASE,
)

# Duration patterns with unit extraction
_RE_DURATION = re.compile(
    r"(?:(\w+(?:\s+(?:and\s+)?(?:a\s+)?(?:half)?)?)\s+"
    r"(days?|weeks?|months?|years?|nights?|hours?))"
    r"|(?:(?:a|an|half\s+a)\s+(day|week|month|year|night|hour))",
    re.IGNORECASE,
)

# Sequential/progressive markers
_RE_NEXT_DAY = re.compile(
    r"(?:^the\s+(?:next|following|previous)\s+day"
    r"|^next\s+day"
    r"|^the\s+day\s+(?:before|after)"
    r"|^that\s+(?:day|evening|morning|night|afternoon)"
    r"|^today$|^tomorrow$|^yesterday$"
    r"|^the\s+(?:first|second|third|fourth|fifth|sixth|seventh)\s+day"
    r"|^the\s+(?:same|very)\s+day"
    r")",
    re.IGNORECASE,
)

# "X later" patterns
_RE_LATER = re.compile(
    r"(\w+(?:\s+\w+)?)\s+(days?|weeks?|months?|years?)\s+later",
    re.IGNORECASE,
)

# Bare day references
_RE_DAY_REF = re.compile(
    r"^(?:the\s+)?day$|^(?:one|One)\s+day$|^days$|^those\s+(?:\w+\s+)?days?$"
    r"|^the\s+(?:final|last|first)\s+days?$",
    re.IGNORECASE,
)

# Bare month/year references
_RE_BARE_MONTH = re.compile(
    r"^(?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December)$",
    re.IGNORECASE,
)
_RE_BARE_WEEKS = re.compile(r"^weeks$|^months$|^years$", re.IGNORECASE)

# Week-level duration
_RE_WEEK_DURATION = re.compile(
    r"(\w+(?:\s+\w+)?)-week$|^(\w+(?:\s+\w+)?)\s+weeks?$", re.IGNORECASE
)
# Month-level duration
_RE_MONTH_DURATION = re.compile(
    r"(\w+(?:\s+\w+)?)-month$|^(\w+(?:\s+\w+)?)\s+months?$", re.IGNORECASE
)
# Year-level duration
_RE_YEAR_DURATION = re.compile(
    r"(\w+(?:\s+\w+)?)-year$|^(\w+(?:\s+\w+)?)\s+years?$", re.IGNORECASE
)
# Day-level duration
_RE_DAY_DURATION = re.compile(
    r"(\w+(?:\s+\w+)?)-day$|^(\w+(?:\s+\w+)?)\s+days?$", re.IGNORECASE
)


def _parse_number(text: str) -> Optional[float]:
    """Parse a number word or digit string."""
    text = text.strip().lower()
    if text in WORD_NUMBERS:
        return WORD_NUMBERS[text]
    # Handle "a half", "and a half"
    if "half" in text:
        parts = text.replace("and", "").replace("a", "").split()
        base = 0.0
        for p in parts:
            if p in WORD_NUMBERS:
                base += WORD_NUMBERS[p]
            elif p == "half":
                base += 0.5
        return base if base > 0 else 0.5
    try:
        return float(text)
    except ValueError:
        return None


def _extract_duration_days(text: str, category: str) -> Optional[float]:
    """Extract an implied duration in days from the expression."""
    clean = re.sub(r"\s+", " ", text).strip().lower()

    unit_days = {"day": 1, "days": 1, "night": 1, "nights": 1,
                 "hour": 0.04, "hours": 0.04,
                 "week": 7, "weeks": 7,
                 "month": 30, "months": 30,
                 "year": 365, "years": 365}

    # "X later" pattern
    m = _RE_LATER.match(clean)
    if m:
        num = _parse_number(m.group(1))
        unit = m.group(2).lower()
        if num and unit in unit_days:
            return num * unit_days[unit]

    # General duration pattern
    m = _RE_DURATION.search(clean)
    if m:
        if m.group(1) and m.group(2):
            num = _parse_number(m.group(1))
            unit = m.group(2).lower()
        elif m.group(3):
            num = 1.0
            unit = m.group(3).lower()
        else:
            return None
        if num and unit in unit_days:
            return num * unit_days[unit]

    # Day-reference defaults
    if category == "day":
        return 1.0
    if category == "week":
        return 7.0
    if category == "month":
        return 30.0
    if category == "year":
        return 365.0
    if category == "season":
        return 90.0

    return None


def classify_temporal_expression(text: str) -> Tuple[str, Optional[float]]:
    """Classify a DATE entity text into a pacing category.

    Returns (category, duration_days) where duration_days is the
    implied temporal span in days, or None if not extractable.
    """
    clean = re.sub(r"\s+", " ", text).strip()

    # --- Noise ---
    if _RE_AGE.match(clean):
        return "noise", None
    if _RE_FRACTION.match(clean):
        return "noise", None
    if len(clean) <= 1:
        return "noise", None

    # --- Calendar years (before prisoner number filter, since 4-digit years overlap) ---
    if _RE_CALENDAR_YEAR.match(clean):
        return "calendar", None

    # --- More noise ---
    if _RE_PRISONER_NUM.match(clean):
        return "noise", None
    if _RE_OCR_NOISE.match(clean):
        return "noise", None

    # --- Habitual ---
    if _RE_HABITUAL.match(clean):
        return "habitual", None

    # --- "X later" (check before duration, gives category from unit) ---
    m = _RE_LATER.match(clean)
    if m:
        unit = m.group(2).lower().rstrip("s")
        cat_map = {"day": "day", "week": "week", "month": "month", "year": "year"}
        cat = cat_map.get(unit, "day")
        dur = _extract_duration_days(clean, cat)
        return cat, dur

    # --- Sequential day references ---
    if _RE_NEXT_DAY.match(clean):
        return "day", 1.0

    # --- Season ---
    if _RE_SEASON.match(clean):
        return "season", 90.0

    # --- Calendar dates (before duration, so "May 1944" doesn't match duration) ---
    if _RE_CALENDAR_RANGE.search(clean):
        return "calendar", None
    if _RE_CALENDAR_FULL.search(clean):
        return "calendar", None
    if _RE_CALENDAR_MONTH_YEAR.search(clean):
        return "calendar", None
    if _RE_CALENDAR_YEAR.match(clean):
        return "calendar", None

    # --- Day-level durations ---
    if _RE_DAY_DURATION.search(clean) and "year" not in clean.lower():
        dur = _extract_duration_days(clean, "day")
        return "day", dur
    if _RE_DAY_REF.match(clean):
        return "day", 1.0

    # --- Week-level durations ---
    if _RE_WEEK_DURATION.search(clean):
        dur = _extract_duration_days(clean, "week")
        return "week", dur

    # --- Month-level durations ---
    if _RE_MONTH_DURATION.search(clean):
        dur = _extract_duration_days(clean, "month")
        return "month", dur

    # --- Year-level durations ---
    if _RE_YEAR_DURATION.search(clean):
        dur = _extract_duration_days(clean, "year")
        return "year", dur

    # --- Bare month names (not duration, treat as calendar) ---
    if _RE_BARE_MONTH.match(clean):
        return "calendar", None

    # --- Bare plurals ---
    if _RE_BARE_WEEKS.match(clean):
        unit = clean.lower().rstrip("s")
        cat_map = {"week": "week", "month": "month", "year": "year"}
        return cat_map.get(unit, "noise"), None

    # --- Fallback: check for any duration-like content ---
    m = _RE_DURATION.search(clean)
    if m:
        unit = (m.group(2) or m.group(3) or "").lower().rstrip("s")
        cat_map = {"day": "day", "night": "day", "hour": "day",
                   "week": "week", "month": "month", "year": "year"}
        cat = cat_map.get(unit, "noise")
        dur = _extract_duration_days(clean, cat)
        return cat, dur

    return "noise", None


# ============================================================================
# Pacing Profile Construction
# ============================================================================

def build_classified_dates(entities_df: pd.DataFrame) -> pd.DataFrame:
    """Classify all DATE entities and add pacing columns."""
    dates = entities_df[entities_df["entity_type"] == "DATE"].copy()
    dates["text_clean"] = dates["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    classifications = dates["text_clean"].apply(classify_temporal_expression)
    dates["category"] = classifications.apply(lambda x: x[0])
    dates["duration_days"] = classifications.apply(lambda x: x[1])

    return dates


def compute_sentence_counts(sentences_df: pd.DataFrame) -> Dict[str, int]:
    """Get total sentence count per document."""
    return (
        sentences_df.groupby("doc_id")["sentence_idx"]
        .max()
        .add(1)
        .to_dict()
    )


def add_text_position(
    dates: pd.DataFrame,
    sentence_counts: Dict[str, int],
) -> pd.DataFrame:
    """Add normalised text position (0 to 1) for each expression."""
    dates = dates.copy()
    dates["total_sentences"] = dates["doc_id"].map(sentence_counts)
    mask = dates["total_sentences"].notna() & (dates["total_sentences"] > 1)
    dates.loc[mask, "text_position"] = (
        dates.loc[mask, "sentence_idx"] / (dates.loc[mask, "total_sentences"] - 1)
    )
    dates.loc[~mask, "text_position"] = 0.0
    dates["text_position"] = dates["text_position"].clip(0.0, 1.0)
    return dates


# ============================================================================
# Per-Document Pacing Metrics
# ============================================================================

def compute_document_pacing(
    dates: pd.DataFrame,
    n_sections: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-document pacing summaries and per-section breakdowns.

    Returns (pacing_by_document, pacing_by_section).
    """
    meaningful = dates[dates["category"] != "noise"].copy()

    # --- Per-document summary ---
    doc_groups = meaningful.groupby("doc_id")

    doc_rows = []
    for doc_id, group in doc_groups:
        n = len(group)
        cats = group["category"].value_counts(normalize=True)
        dominant = cats.index[0] if len(cats) > 0 else None
        durations = group["duration_days"].dropna()

        row = {
            "doc_id": doc_id,
            "n_expressions": n,
            "dominant_category": dominant,
            "median_duration_days": durations.median() if len(durations) > 0 else None,
            "mean_duration_days": durations.mean() if len(durations) > 0 else None,
        }
        for cat in PACING_CATEGORIES:
            if cat != "noise":
                row[f"frac_{cat}"] = cats.get(cat, 0.0)
        doc_rows.append(row)

    pacing_by_doc = pd.DataFrame(doc_rows)

    # --- Per-section breakdown ---
    meaningful["section"] = pd.cut(
        meaningful["text_position"],
        bins=n_sections,
        labels=[f"section_{i+1}" for i in range(n_sections)],
        include_lowest=True,
    )

    section_rows = []
    for (doc_id, section), group in meaningful.groupby(["doc_id", "section"], observed=True):
        n = len(group)
        cats = group["category"].value_counts(normalize=True)
        dominant = cats.index[0] if len(cats) > 0 else None
        durations = group["duration_days"].dropna()

        row = {
            "doc_id": doc_id,
            "section": section,
            "n_expressions": n,
            "dominant_category": dominant,
            "median_duration_days": durations.median() if len(durations) > 0 else None,
        }
        for cat in PACING_CATEGORIES:
            if cat != "noise":
                row[f"frac_{cat}"] = cats.get(cat, 0.0)
        section_rows.append(row)

    pacing_by_section = pd.DataFrame(section_rows)

    return pacing_by_doc, pacing_by_section


# ============================================================================
# Corpus-Level Aggregation
# ============================================================================

def compute_corpus_pacing_by_position(
    dates: pd.DataFrame,
    n_bins: int = 20,
) -> pd.DataFrame:
    """Compute category distribution as a function of text position.

    Divides text position into n_bins and computes the fraction of
    each category in each bin, averaged across documents.
    """
    meaningful = dates[dates["category"] != "noise"].copy()
    meaningful["position_bin"] = pd.cut(
        meaningful["text_position"],
        bins=n_bins,
        labels=False,
        include_lowest=True,
    )
    meaningful["position_mid"] = (meaningful["position_bin"] + 0.5) / n_bins

    rows = []
    for pos_bin, group in meaningful.groupby("position_bin", observed=True):
        cats = group["category"].value_counts(normalize=True)
        row = {
            "position_bin": pos_bin,
            "position_mid": (pos_bin + 0.5) / n_bins,
            "n_expressions": len(group),
            "n_documents": group["doc_id"].nunique(),
        }
        for cat in PACING_CATEGORIES:
            if cat != "noise":
                row[f"frac_{cat}"] = cats.get(cat, 0.0)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("position_bin")


def build_report(
    classified: pd.DataFrame,
    pacing_by_doc: pd.DataFrame,
    n_sections: int,
) -> dict:
    """Build a JSON-serialisable summary report."""
    total = len(classified)
    cat_counts = classified["category"].value_counts()
    noise_count = cat_counts.get("noise", 0)
    meaningful_count = total - noise_count

    report = {
        "total_date_entities": total,
        "meaningful_expressions": int(meaningful_count),
        "noise_filtered": int(noise_count),
        "noise_fraction": round(noise_count / total, 3) if total > 0 else 0,
        "documents_with_expressions": int(pacing_by_doc["n_expressions"].gt(0).sum()),
        "n_sections": n_sections,
        "category_counts": {
            cat: int(cat_counts.get(cat, 0)) for cat in PACING_CATEGORIES
        },
        "category_fractions": {
            cat: round(cat_counts.get(cat, 0) / total, 3) if total > 0 else 0
            for cat in PACING_CATEGORIES
        },
        "median_expressions_per_doc": round(float(pacing_by_doc["n_expressions"].median()), 1),
        "corpus_dominant_categories": (
            pacing_by_doc["dominant_category"]
            .value_counts()
            .head(5)
            .to_dict()
        ),
    }
    return report


# ============================================================================
# Visualisation
# ============================================================================

def plot_category_distribution(
    classified: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Bar chart of overall pacing category distribution."""
    meaningful = classified[classified["category"] != "noise"]
    cats = meaningful["category"].value_counts()

    ordered = [c for c in PACING_CATEGORIES if c != "noise" and c in cats.index]
    counts = [cats[c] for c in ordered]
    colours = [CATEGORY_COLOURS[c] for c in ordered]
    labels = [CATEGORY_LABELS[c] for c in ordered]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, counts, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Count")
    ax.set_title("Temporal Expression Categories in Testimonies")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            f"{count:,}", ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    path = output_dir / "category_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_pacing_ribbon(
    corpus_by_position: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Stacked area chart showing how pacing category mix changes
    from beginning to end of testimonies."""
    display_cats = [c for c in ["day", "week", "month", "year", "habitual", "calendar", "season"]
                    if f"frac_{c}" in corpus_by_position.columns]

    x = corpus_by_position["position_mid"].values
    ys = np.array([corpus_by_position[f"frac_{c}"].values for c in display_cats])
    colours = [CATEGORY_COLOURS[c] for c in display_cats]
    labels = [CATEGORY_LABELS[c] for c in display_cats]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(x, ys, labels=labels, colors=colours, alpha=0.85)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Text Position (beginning to end)")
    ax.set_ylabel("Fraction of Temporal Expressions")
    ax.set_title("Narrative Pacing Through Testimonies (Corpus Average)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = output_dir / "pacing_ribbon.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_pacing_by_section(
    pacing_by_section: pd.DataFrame,
    n_sections: int,
    output_dir: Path,
) -> Path:
    """Grouped bar chart comparing category distributions across
    testimony sections (beginning, middle, end)."""
    display_cats = ["day", "week", "month", "year", "habitual", "calendar", "season"]
    section_labels = {
        3: ["Beginning", "Middle", "End"],
        5: ["1st fifth", "2nd fifth", "3rd fifth", "4th fifth", "5th fifth"],
    }
    labels = section_labels.get(n_sections, [f"Section {i+1}" for i in range(n_sections)])

    sections = sorted(pacing_by_section["section"].unique())
    section_means = {}
    for section in sections:
        mask = pacing_by_section["section"] == section
        section_means[section] = {
            cat: pacing_by_section.loc[mask, f"frac_{cat}"].mean()
            for cat in display_cats
            if f"frac_{cat}" in pacing_by_section.columns
        }

    fig, ax = plt.subplots(figsize=(12, 6))
    n_cats = len(display_cats)
    n_secs = len(sections)
    width = 0.8 / n_secs
    x = np.arange(n_cats)

    for i, (section, label) in enumerate(zip(sections, labels)):
        vals = [section_means[section].get(cat, 0) for cat in display_cats]
        offset = (i - n_secs / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width,
            label=label, alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in display_cats], rotation=15)
    ax.set_ylabel("Mean Fraction")
    ax.set_title("Pacing by Testimony Section")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = output_dir / "pacing_by_section.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_pacing_strips(
    classified: pd.DataFrame,
    output_dir: Path,
    n_testimonies: int = 30,
) -> Path:
    """Horizontal barcode-style strips showing the pacing category
    at each temporal expression's text position, stacked vertically
    for comparison across testimonies."""
    meaningful = classified[classified["category"] != "noise"]
    doc_counts = meaningful.groupby("doc_id").size()
    top_docs = doc_counts.nlargest(n_testimonies).index.tolist()

    fig, ax = plt.subplots(figsize=(14, max(6, n_testimonies * 0.3)))

    for i, doc_id in enumerate(top_docs):
        doc_data = meaningful[meaningful["doc_id"] == doc_id].sort_values("text_position")
        for _, row in doc_data.iterrows():
            colour = CATEGORY_COLOURS.get(row["category"], "#999999")
            ax.scatter(
                row["text_position"], i,
                c=colour, s=12, marker="|", linewidths=1.5,
            )

    ax.set_yticks(range(len(top_docs)))
    ax.set_yticklabels([f"Doc {d}" for d in top_docs], fontsize=7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Text Position (beginning to end)")
    ax.set_title(f"Pacing Strips: Top {n_testimonies} Testimonies by Expression Count")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], color=CATEGORY_COLOURS[c], linewidth=3, label=CATEGORY_LABELS[c])
        for c in ["day", "week", "month", "year", "habitual", "calendar", "season"]
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    path = output_dir / "pacing_strips.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_duration_distribution(
    classified: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Histogram of implied durations (in days) across all expressions."""
    meaningful = classified[
        (classified["category"] != "noise")
        & classified["duration_days"].notna()
        & (classified["duration_days"] > 0)
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = [0, 1, 2, 3, 7, 14, 30, 60, 90, 180, 365, 730, 1825]
    bin_labels = ["<1d", "1d", "2d", "3-6d", "1-2w", "2w-1m", "1-2m", "2-3m", "3-6m", "6m-1y", "1-2y", "2-5y"]
    counts, _ = np.histogram(meaningful["duration_days"], bins=bins)

    x = np.arange(len(bin_labels))
    colours = []
    for b in bins[:-1]:
        if b < 3:
            colours.append(CATEGORY_COLOURS["day"])
        elif b < 14:
            colours.append(CATEGORY_COLOURS["week"])
        elif b < 90:
            colours.append(CATEGORY_COLOURS["month"])
        else:
            colours.append(CATEGORY_COLOURS["year"])

    ax.bar(x, counts, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=30)
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Implied Temporal Durations")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = output_dir / "duration_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Narrative pacing analysis of Ravensbrück testimonies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--entities", "-e", type=Path, default=DEFAULT_ENTITIES,
        help=f"Input entities CSV (default: {DEFAULT_ENTITIES.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--sentences", "-s", type=Path, default=DEFAULT_SENTENCES,
        help=f"Input sentences CSV (default: {DEFAULT_SENTENCES.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--sections", type=int, default=3,
        help="Number of sections to divide testimonies into (default: 3)",
    )
    parser.add_argument(
        "--position-bins", type=int, default=20,
        help="Number of bins for position-based analysis (default: 20)",
    )
    parser.add_argument(
        "--strip-count", type=int, default=30,
        help="Number of testimonies in pacing strip plot (default: 30)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    for path in [args.entities, args.sentences]:
        if not path.exists():
            logger.error(f"Input file not found: {path}")
            sys.exit(1)

    # ---- Banner ----
    logger.info("=" * 60)
    logger.info("NARRATIVE PACING ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Entities:  {args.entities}")
    logger.info(f"Sentences: {args.sentences}")
    logger.info(f"Output:    {args.output}")
    logger.info(f"Sections:  {args.sections}")

    # ---- Load ----
    logger.info("Loading data...")
    entities_df = pd.read_csv(args.entities)
    sentences_df = pd.read_csv(args.sentences, usecols=["doc_id", "sentence_idx"])
    sentence_counts = compute_sentence_counts(sentences_df)
    logger.info(f"  {len(entities_df):,} entities across {len(sentence_counts)} documents")

    # ---- Classify ----
    logger.info("Classifying temporal expressions...")
    classified = build_classified_dates(entities_df)
    classified = add_text_position(classified, sentence_counts)
    cat_counts = classified["category"].value_counts()
    for cat in PACING_CATEGORIES:
        count = cat_counts.get(cat, 0)
        pct = 100 * count / len(classified) if len(classified) > 0 else 0
        logger.info(f"  {cat:10s}: {count:5,} ({pct:5.1f}%)")

    # ---- Document pacing ----
    logger.info("Computing per-document pacing metrics...")
    pacing_by_doc, pacing_by_section = compute_document_pacing(
        classified, n_sections=args.sections,
    )
    logger.info(f"  {len(pacing_by_doc)} documents with pacing profiles")

    # ---- Corpus position analysis ----
    logger.info("Computing corpus pacing by text position...")
    corpus_by_position = compute_corpus_pacing_by_position(
        classified, n_bins=args.position_bins,
    )

    # ---- Report ----
    report = build_report(classified, pacing_by_doc, args.sections)

    # ---- Save outputs ----
    args.output.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving data outputs...")
    classified.to_parquet(args.output / "classified_dates.parquet", index=False)
    pacing_by_doc.to_parquet(args.output / "pacing_by_document.parquet", index=False)
    pacing_by_section.to_parquet(args.output / "pacing_by_section.parquet", index=False)
    corpus_by_position.to_parquet(args.output / "pacing_by_position.parquet", index=False)

    report_path = args.output / "pacing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"  Report: {report_path.name}")

    # ---- Visualise ----
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING FIGURES")
    logger.info("=" * 60)

    p = plot_category_distribution(classified, fig_dir)
    logger.info(f"  Saved {p.name}")

    p = plot_pacing_ribbon(corpus_by_position, fig_dir)
    logger.info(f"  Saved {p.name}")

    p = plot_pacing_by_section(pacing_by_section, args.sections, fig_dir)
    logger.info(f"  Saved {p.name}")

    p = plot_pacing_strips(classified, fig_dir, n_testimonies=args.strip_count)
    logger.info(f"  Saved {p.name}")

    p = plot_duration_distribution(classified, fig_dir)
    logger.info(f"  Saved {p.name}")

    # ---- Done ----
    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
