"""Co-variation between testimony metadata and word categories.

For each testimony, counts how often the terms of a category (from the
project Google Sheet) occur in the testimony body, and relates that to a
metadata variable (gender, religion, age, interviewer, ...).

Measures per testimony:
    hits        number of term occurrences
    rate        hits per 1,000 words (corrects for testimony length)
    mentions    whether the testimony contains at least one term

Tests:
    group variables    chi-square on `mentions` (effect: Cramér's V) and
                       Kruskal–Wallis on `rate` (effect: epsilon²)
    numeric variables  Spearman correlation with `rate` (effect: rho), plus
                       the same group tables on binned values

With --category all and/or --variable all, writes one overview row per
category × variable, with Benjamini–Hochberg adjusted p-values.

Usage:
    python 13_category_stats.py --category Time --variable gender
    python 13_category_stats.py --category Emotions --variable all
    python 13_category_stats.py --category all --variable all --gender female
    python 13_category_stats.py --category Time --variable recorder \\
        --output-gsheet OUTPUT_SHEET_ID --credentials creds.json

Requirements:
    pandas, pyarrow, requests, scipy
    gspread, google-auth (only for Google Sheets output)
"""

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from sheet_categories import (
    DEFAULT_SHEET_ID,
    fetch_categories,
    open_spreadsheet,
    term_pattern,
    write_tab,
)


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "metadata.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "stats"

TEXT_COL = "testimony_body"
TOP_LEVELS = 6          # keep the N most common levels, pool the rest as "Other"
MIN_GROUP_SIZE = 5      # smaller groups are pooled into "Other" / dropped
TOP_TERMS = 25          # rows in the per-term table

ABOUT_ROWS = [
    ["How to read this"],
    ["rate = matches per 1,000 words (corrects for testimony length)"],
    ["% mentioning = share of testimonies with at least one match"],
    ["p < 0.05 is conventionally 'significant'; with many tests, use p (BH-adjusted)"],
    ["Effect sizes: Cramér's V / |rho| ≈ 0.1 small, 0.3 medium, 0.5 large; "
     "epsilon² ≈ 0.01 small, 0.06 medium, 0.14 large"],
    ["Small groups (<5 testimonies) are pooled or dropped; results describe "
     "this corpus, not causes"],
]

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================================
# Metadata variables
# ============================================================================

def year_from(text: object) -> float:
    """First plausible 4-digit year (1850–1950) in a free-text date."""
    m = re.search(r"\b(18[5-9]\d|19[0-4]\d|1950)\b", str(text))
    return float(m.group(1)) if m else np.nan


def pool_levels(values: pd.Series, top: int = TOP_LEVELS) -> pd.Series:
    """Keep the most common levels (with enough cases); pool the rest."""
    counts = values.value_counts()
    keep = counts[counts >= MIN_GROUP_SIZE].index[:top]
    return values.where(values.isin(keep) | values.isna(), "Other")


def clean_religion(value: object) -> Optional[str]:
    text = str(value).lower()
    if "catholic" in text:
        return "Roman Catholic"
    if "jew" in text or "mosaic" in text:
        return "Jewish"
    if text in ("", "none", "nan") or "not completed" in text:
        return None
    return "Other"


def testimony_dates(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["testimony_date"], errors="coerce", dayfirst=True, format="mixed")


@dataclass
class Variable:
    name: str
    label: str
    kind: str                                   # "group" or "numeric"
    build: Callable[[pd.DataFrame], pd.Series]
    bins: Optional[List[float]] = None          # numeric only: bins for group tables


def _age(df: pd.DataFrame) -> pd.Series:
    return testimony_dates(df).dt.year - df["birth_date"].map(year_from)


def _months_since_start(df: pd.DataFrame) -> pd.Series:
    dates = testimony_dates(df)
    start = dates.min()
    return (dates.dt.year - start.year) * 12 + (dates.dt.month - start.month)


VARIABLES: Dict[str, Variable] = {v.name: v for v in [
    Variable("gender", "Gender", "group",
             lambda df: df["gender"].where(df["gender"].isin(["female", "male"]))),
    Variable("religion", "Religion", "group",
             lambda df: pool_levels(df["religion"].map(clean_religion))),
    Variable("age", "Age at testimony (years)", "numeric", _age,
             bins=[0, 20, 30, 40, 50, 120]),
    Variable("testimony_date", "Testimony date (months since first testimony)",
             "numeric", _months_since_start, bins=[-1, 2, 5, 8, 11, 60]),
    Variable("testimony_place", "Place of testimony", "group",
             lambda df: pool_levels(df["testimony_place"].replace("", None))),
    Variable("recorder", "Interviewer", "group",
             lambda df: pool_levels(df["recorder_name"].replace("", None), top=10)),
    Variable("series", "Series (regular / confidential)", "group",
             lambda df: df["is_confidential"].map({True: "confidential", False: "regular"})),
    Variable("length", "Testimony length (words)", "numeric",
             lambda df: df["n_words"].astype(float),
             bins=[0, 1000, 2000, 4000, 8000, 1e9]),
]}


# ============================================================================
# Counting
# ============================================================================

def count_terms(texts: pd.Series, terms: List[str]) -> pd.DataFrame:
    """Return a documents × terms matrix of occurrence counts."""
    patterns = {t: term_pattern(t) for t in terms}
    return pd.DataFrame(
        {t: texts.map(lambda s, p=p: len(p.findall(s))) for t, p in patterns.items()},
        index=texts.index,
    )


def doc_measures(term_counts: pd.DataFrame, n_words: pd.Series) -> pd.DataFrame:
    hits = term_counts.sum(axis=1)
    return pd.DataFrame({
        "hits": hits,
        "rate": hits / n_words * 1000,
        "mentions": hits > 0,
    })


# ============================================================================
# Statistics
# ============================================================================

def grouping_for(var: Variable, values: pd.Series) -> pd.Series:
    """Group labels for tables: the variable itself, or binned numeric values."""
    if var.kind == "group":
        return values
    labels = []
    for lo, hi in zip(var.bins[:-1], var.bins[1:]):
        if var.name == "length":
            labels.append(f"{int(lo):,}–{int(hi):,}" if hi < 1e8 else f"{int(lo):,}+")
        else:
            labels.append(f"{lo:g}–{hi:g}" if hi < 100 else f"{lo:g}+")
    return pd.cut(values, bins=var.bins, labels=labels, right=False).astype(object)


def group_table(measures: pd.DataFrame, groups: pd.Series, n_words: pd.Series) -> pd.DataFrame:
    d = measures.assign(group=groups, n_words=n_words).dropna(subset=["group"])
    table = d.groupby("group").agg(
        testimonies=("hits", "size"),
        pct_mentioning=("mentions", "mean"),
        mean_rate=("rate", "mean"),
        median_rate=("rate", "median"),
        total_hits=("hits", "sum"),
        total_words=("n_words", "sum"),
    )
    table["pooled_rate"] = table["total_hits"] / table["total_words"] * 1000
    table["pct_mentioning"] *= 100
    table = table[table["testimonies"] >= MIN_GROUP_SIZE]
    return table.drop(columns="total_words").round(2).reset_index()


def term_table(term_counts: pd.DataFrame, groups: pd.Series, n_words: pd.Series) -> pd.DataFrame:
    """Pooled rate per 1,000 words for each term within each group."""
    valid = groups.notna()
    counts, groups, n_words = term_counts[valid], groups[valid], n_words[valid]
    sizes = groups.value_counts()
    keep = sizes[sizes >= MIN_GROUP_SIZE].index
    top = counts.sum().sort_values(ascending=False)
    top = top[top > 0].index[:TOP_TERMS]

    words_by_group = n_words.groupby(groups).sum()
    rates = counts[top].groupby(groups).sum().div(words_by_group, axis=0) * 1000
    rates = rates.loc[[g for g in rates.index if g in keep]].T.round(3)
    rates.insert(0, "total_hits", counts[top].sum())
    return rates.reset_index().rename(columns={"index": "term"})


def run_tests(var: Variable, values: pd.Series, measures: pd.DataFrame) -> dict:
    """Return a dict with test name, statistic, p-value, effect size and n."""
    d = measures.assign(x=values).dropna(subset=["x"])
    result = {"variable": var.name, "n": len(d)}

    if var.kind == "numeric":
        rho, p = stats.spearmanr(d["x"], d["rate"])
        result.update(test="Spearman (rate)", statistic=rho, p=p,
                      effect="rho", effect_size=rho, note="")
        return result

    sizes = d["x"].value_counts()
    d = d[d["x"].isin(sizes[sizes >= MIN_GROUP_SIZE].index)]
    k = d["x"].nunique()
    if k < 2:
        result.update(test="—", statistic=np.nan, p=np.nan, effect="", effect_size=np.nan,
                      note="fewer than two groups with enough testimonies")
        return result

    h, p_kw = stats.kruskal(*[g["rate"].values for _, g in d.groupby("x")])
    epsilon2 = max((h - k + 1) / (len(d) - k), 0.0)

    contingency = pd.crosstab(d["x"], d["mentions"])
    note = ""
    if contingency.shape[1] < 2:
        chi2, p_chi, cramers_v = np.nan, np.nan, np.nan
        note = "all or no testimonies mention the category"
    else:
        chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
        cramers_v = np.sqrt(chi2 / (contingency.values.sum() * (min(contingency.shape) - 1)))
        if (expected < 5).any():
            note = "some expected counts < 5; chi-square approximate"

    result.update(test="Kruskal–Wallis (rate)", statistic=h, p=p_kw,
                  effect="epsilon²", effect_size=epsilon2,
                  chi2_mentions=chi2, p_chi2=p_chi, cramers_v=cramers_v, note=note)
    return result


def bh_adjust(p: pd.Series) -> pd.Series:
    """Benjamini–Hochberg adjusted p-values (NaNs left as NaN)."""
    valid = p.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return p
    adjusted = (valid * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1)
    return adjusted.reindex(p.index)


# ============================================================================
# Output formatting
# ============================================================================

def df_rows(df: pd.DataFrame) -> List[list]:
    """DataFrame -> header + rows, with NaN as empty cells (Sheets-safe)."""
    clean = df.astype(object).where(df.notna(), "")
    return [list(map(str, df.columns))] + clean.values.tolist()


def tests_frame(results: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df["p (BH-adjusted)"] = bh_adjust(df["p"])
    if "p_chi2" in df:
        df["p_chi2 (BH-adjusted)"] = bh_adjust(df["p_chi2"])
    ordered = [c for c in ["category", "variable", "n", "test", "statistic", "p",
                           "p (BH-adjusted)", "effect", "effect_size", "chi2_mentions",
                           "p_chi2", "p_chi2 (BH-adjusted)", "cramers_v", "note"]
               if c in df.columns]
    df = df[ordered]
    for col in df.columns:
        if col.startswith("p"):
            df[col] = df[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    return df.round(3)


def slug(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text.lower()).strip("_")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test co-variation between metadata and word categories.",
    )
    parser.add_argument("--sheet-id", default=DEFAULT_SHEET_ID,
                        help="Google Sheet with the word categories")
    parser.add_argument("--category", "-c", required=True,
                        help="Category (column header) or 'all'")
    parser.add_argument("--variable", default="all",
                        help=f"Metadata variable or 'all': {', '.join(VARIABLES)}")
    parser.add_argument("--gender", choices=["female", "male"], default=None,
                        help="Restrict to one gender before testing")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-gsheet", default=None,
                        help="Google Sheet ID to write results to (one tab per run)")
    parser.add_argument("--credentials", default=None,
                        help="Service account JSON (or set GOOGLE_CREDENTIALS)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    # -- Categories and variables ------------------------------------------
    categories = fetch_categories(args.sheet_id)
    if args.category.lower() == "all":
        selected_cats = list(categories)
    elif args.category in categories:
        selected_cats = [args.category]
    else:
        logger.error("Category '%s' not found. Available: %s", args.category, list(categories))
        sys.exit(1)

    if args.variable.lower() == "all":
        selected_vars = list(VARIABLES.values())
    elif args.variable in VARIABLES:
        selected_vars = [VARIABLES[args.variable]]
    else:
        logger.error("Variable '%s' not found. Available: %s", args.variable, list(VARIABLES))
        sys.exit(1)
    if args.gender:
        selected_vars = [v for v in selected_vars if v.name != "gender"]

    # -- Corpus -------------------------------------------------------------
    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)
    df = pd.read_parquet(args.input)
    df = df[df[TEXT_COL].fillna("").str.strip().astype(bool)].copy()
    if args.gender:
        df = df[df["gender"] == args.gender]
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df["n_words"] = df[TEXT_COL].str.split().str.len()
    logger.info("Analysing %d testimonies%s", len(df),
                f" ({args.gender} only)" if args.gender else "")

    variable_values = {v.name: v.build(df) for v in selected_vars}

    # -- Analyse ------------------------------------------------------------
    results: List[dict] = []
    detail: Dict[str, pd.DataFrame] = {}
    for cat in selected_cats:
        counts = count_terms(df[TEXT_COL], categories[cat])
        measures = doc_measures(counts, df["n_words"])
        logger.info("%s: %d hits; %.0f%% of testimonies mention it",
                    cat, measures["hits"].sum(), measures["mentions"].mean() * 100)
        for var in selected_vars:
            values = variable_values[var.name]
            results.append({"category": cat, **run_tests(var, values, measures)})
            if len(selected_cats) == 1 and len(selected_vars) == 1:
                groups = grouping_for(var, values)
                detail["groups"] = group_table(measures, groups, df["n_words"])
                detail["terms"] = term_table(counts, groups, df["n_words"])

    tests = tests_frame(results)
    if len(selected_cats) == 1:
        tests = tests.drop(columns="category")

    # -- Report -------------------------------------------------------------
    scope = f"{args.category} × {args.variable}" + (f" ({args.gender})" if args.gender else "")
    print(f"\n=== {scope} ===\n")
    print(tests.to_string(index=False))
    for name, table in detail.items():
        print(f"\n--- {name} ---\n{table.to_string(index=False)}")

    blocks: List[list] = [[f"Category statistics: {scope}"],
                          [f"{len(df)} testimonies; terms from the category sheet"], []]
    blocks += [["Tests"]] + df_rows(tests) + [[]]
    if "groups" in detail:
        var = selected_vars[0]
        blocks += [[f"By {var.label.lower()}"]] + df_rows(detail["groups"]) + [[]]
        blocks += [["Rate per 1,000 words by term (top terms)"]] + df_rows(detail["terms"]) + [[]]
    blocks += ABOUT_ROWS

    if args.output_gsheet:
        spreadsheet = open_spreadsheet(args.output_gsheet, args.credentials)
        write_tab(spreadsheet, f"Stats: {scope}", blocks, freeze_rows=0)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        base = slug(scope.replace("×", "x"))
        tests.to_csv(args.output_dir / f"{base}_tests.csv", index=False)
        for name, table in detail.items():
            table.to_csv(args.output_dir / f"{base}_{name}.csv", index=False)
        logger.info("Saved results to %s (%s_*.csv)", args.output_dir, base)


if __name__ == "__main__":
    main()
