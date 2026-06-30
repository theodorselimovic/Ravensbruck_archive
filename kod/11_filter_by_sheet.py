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
    pandas, pyarrow, spacy, requests
    gspread, google-auth (only for Google Sheets output)
"""

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "metadata.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "filtered"
DEFAULT_SHEET_ID = "1qhmuSIxo_kpRudFiyVenyr4BsvoNH4NVoVbzJELnQ5g"


# ============================================================================
# Google Sheet fetcher
# ============================================================================

def fetch_categories(sheet_id: str) -> Dict[str, List[str]]:
    """Fetch a public Google Sheet and return {category_name: [terms]}.

    Each cell is split on commas (for explicit alternatives).
    Terms are searched exactly as written.
    """
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    sheet = pd.read_csv(io.StringIO(resp.text))
    categories: Dict[str, List[str]] = {}

    for col in sheet.columns:
        seen: set = set()
        terms: List[str] = []

        for val in sheet[col].dropna():
            for term in str(val).split(","):
                term = term.strip()
                if not term:
                    continue
                key = term.lower()
                if key not in seen:
                    seen.add(key)
                    terms.append(term)

        if terms:
            categories[col] = terms

    return categories


# ============================================================================
# Sentencizer
# ============================================================================

SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

def sentencize(text: str) -> List[str]:
    """Split text into sentences using punctuation + capital letter boundaries."""
    if not text or not text.strip():
        return []
    return [s.strip() for s in SENT_BOUNDARY.split(text) if s.strip()]


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
    """Write results to a tab in a Google Sheet."""
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
    ]

    creds_json = os.environ.get("GOOGLE_CREDENTIALS")
    if creds_json:
        info = json.loads(creds_json)
        creds = Credentials.from_service_account_info(info, scopes=scopes)
    elif credentials_path:
        creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
    else:
        print(
            "No credentials provided. Use --credentials or set GOOGLE_CREDENTIALS env var.",
            file=sys.stderr,
        )
        sys.exit(1)

    gc = gspread.authorize(creds)
    spreadsheet = gc.open_by_key(sheet_id)

    tab_name = category
    if gender:
        tab_name += f" ({gender})"

    try:
        worksheet = spreadsheet.worksheet(tab_name)
        worksheet.clear()
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=tab_name, rows=len(result) + 1, cols=4)

    data = [result.columns.tolist()] + result.values.tolist()
    worksheet.update(range_name="A1", values=data)

    print(f"Wrote {len(result)} rows to Google Sheet tab '{tab_name}'")


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

    if "testimony_number" not in df.columns:
        df["testimony_number"] = df.index.astype(str)

    if args.gender:
        if "gender" not in df.columns:
            print("No gender column found; cannot filter by gender.", file=sys.stderr)
            sys.exit(1)
        df = df[df["gender"].str.lower() == args.gender.lower()]
        print(f"Filtered to {len(df)} {args.gender} documents")

    # -- Build patterns (bigrams first so longest match wins) ----------------
    patterns = [(w, re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE)) for w in words]

    # -- Search sentences ---------------------------------------------------
    rows: List[dict] = []

    for _, doc_row in df.iterrows():
        raw_id = doc_row["testimony_number"]
        if pd.isna(raw_id):
            doc_id = "unknown"
        elif isinstance(raw_id, float) and raw_id == int(raw_id):
            doc_id = str(int(raw_id))
        else:
            doc_id = str(raw_id)

        text = doc_row[text_col]
        if not text or not isinstance(text, str):
            continue

        sentences = sentencize(text)
        for sent_idx, sent in enumerate(sentences):
            for word, pat in patterns:
                if pat.search(sent):
                    rows.append({
                        "doc_id": doc_id,
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
