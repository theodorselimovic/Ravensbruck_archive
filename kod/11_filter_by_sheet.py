"""Filter corpus by word categories from a Google Sheet.

Fetches a public Google Sheet where each column is a category (header = name,
rows = search terms). Searches testimony text at the sentence level for any
word in the chosen category and exports matching sentences with the matched
word, sentence index, and document ID.

Results can be saved locally as CSV or written back to a Google Sheet
(requires a service account credentials JSON).

Usage:
    python 11_filter_by_sheet.py                         # interactive category picker
    python 11_filter_by_sheet.py --category "Confined space"
    python 11_filter_by_sheet.py --category "Confined space" --gender female
    python 11_filter_by_sheet.py --output-gsheet OUTPUT_SHEET_ID --credentials creds.json

Requirements:
    pandas, pyarrow, requests
    gspread, google-auth (only for Google Sheets output)
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List

import pandas as pd

from sheet_categories import (
    DEFAULT_SHEET_ID,
    fetch_categories,
    open_spreadsheet,
    sentencize,
    term_pattern,
    write_tab,
)


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "metadata.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "filtered"


# ============================================================================
# Helpers
# ============================================================================

def format_testimony_number(raw: object) -> str:
    """Render the printed testimony number (e.g. 3.0 -> "3"); "" if missing."""
    if raw is None or pd.isna(raw):
        return ""
    if isinstance(raw, float) and raw == int(raw):
        return str(int(raw))
    return str(raw)


# ============================================================================
# Google Sheet writer
# ============================================================================

def write_to_gsheet(
    result: pd.DataFrame,
    sheet_id: str,
    category: str,
    credentials_path: str | None,
    gender: str | None,
) -> None:
    """Write results to a tab (named after the category) in a Google Sheet."""
    tab_name = category + (f" ({gender})" if gender else "")
    rows = [result.columns.tolist()] + result.values.tolist()
    write_tab(open_spreadsheet(sheet_id, credentials_path), tab_name, rows)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter corpus by word categories from a Google Sheet.",
    )
    parser.add_argument(
        "--sheet-id",
        type=str,
        default=DEFAULT_SHEET_ID,
        help="Google Sheet ID (default: project sheet)",
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        help="Category name (column header) to search. If omitted, shows a picker.",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input parquet file (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for CSV files (default: %(default)s)",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default=None,
        choices=["female", "male"],
        help="Filter corpus by gender before searching",
    )
    parser.add_argument(
        "--output-gsheet",
        type=str,
        default=None,
        help="Google Sheet ID to write results to (creates/overwrites a tab named after the category)",
    )
    parser.add_argument(
        "--credentials",
        type=str,
        default=None,
        help="Path to service account JSON, or set GOOGLE_CREDENTIALS env var with the JSON content",
    )
    args = parser.parse_args()

    # -- Fetch categories from sheet ----------------------------------------
    print("Fetching categories from Google Sheet...")
    categories = fetch_categories(args.sheet_id)

    if not categories:
        print("No categories found in the sheet.", file=sys.stderr)
        sys.exit(1)

    # -- Select category ----------------------------------------------------
    if args.category:
        if args.category not in categories:
            print(
                f"Category '{args.category}' not found. "
                f"Available: {list(categories.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)
        selected = args.category
    else:
        print("\nAvailable categories:")
        names = list(categories.keys())
        for idx, name in enumerate(names, 1):
            print(f"  [{idx}] {name} ({len(categories[name])} words)")
        choice = input("\nSelect category number: ").strip()
        try:
            selected = names[int(choice) - 1]
        except (ValueError, IndexError):
            print("Invalid selection.", file=sys.stderr)
            sys.exit(1)

    words = categories[selected]
    print(f"\nCategory: {selected}")
    print(f"Search terms ({len(words)}): {', '.join(words)}")

    # -- Load corpus --------------------------------------------------------
    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(args.input)

    text_col = "testimony_body" if "testimony_body" in df.columns else "testimony"
    if text_col not in df.columns:
        print(f"No text column found. Columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # doc_id is the filename stem (e.g. "3" vs "t-3"); testimony_number is
    # NOT unique across files, so it must not be used as an identifier.
    if "doc_id" not in df.columns:
        df["doc_id"] = df.index.astype(str)

    if args.gender:
        if "gender" not in df.columns:
            print("No gender column found; cannot filter by gender.", file=sys.stderr)
            sys.exit(1)
        df = df[df["gender"].str.lower() == args.gender.lower()]
        print(f"Filtered to {len(df)} {args.gender} documents")

    # -- Build patterns (bigrams first so longest match wins) ----------------
    patterns = [(w, term_pattern(w)) for w in words]

    # -- Search sentences ---------------------------------------------------
    rows: List[dict] = []

    for _, doc_row in df.iterrows():
        doc_id = str(doc_row["doc_id"])
        testimony_number = format_testimony_number(doc_row.get("testimony_number"))

        text = doc_row[text_col]
        if not text or not isinstance(text, str):
            continue

        sentences = sentencize(text)
        for sent_idx, sent in enumerate(sentences):
            for word, pat in patterns:
                if pat.search(sent):
                    rows.append({
                        "doc_id": doc_id,
                        "testimony_number": testimony_number,
                        "sentence_idx": sent_idx,
                        "matched_word": word,
                        "sentence": sent,
                    })

    if not rows:
        print(f"No sentences found for category '{selected}'.")
        sys.exit(0)

    result = pd.DataFrame(rows)
    print(f"\nFound {len(result)} matching sentences.")

    # -- Save to Google Sheet -----------------------------------------------
    if args.output_gsheet:
        write_to_gsheet(result, args.output_gsheet, selected, args.credentials, args.gender)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^\w\-]", "_", selected.lower())
        parts = [safe_name]
        if args.gender:
            parts.append(args.gender)
        filename = "_".join(parts) + ".csv"
        output_path = args.output_dir / filename
        result.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
