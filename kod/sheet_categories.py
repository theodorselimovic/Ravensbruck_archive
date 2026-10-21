"""Shared helpers for the Google Sheet of word categories.

Used by 11_filter_by_sheet.py and 13_category_stats.py:

    fetch_categories(sheet_id)   {category: [terms]} from the public sheet
    term_pattern(term)           compiled whole-word pattern for a word/phrase
    sentencize(text)             lightweight sentence splitter
    open_spreadsheet(...)        authorised gspread Spreadsheet
    write_tab(...)               (re)write one tab with a list of rows

Sheet format: each column is a category (header = name), each cell holds one
or more comma-separated terms. Phrases may be wrapped in quotes
("camp mother", “white bus”); the quotes are removed and commas inside them
do not split the phrase.
"""

import io
import json
import os
import re
import sys
from typing import Dict, List, Optional

import pandas as pd
import requests


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_SHEET_ID = "1qhmuSIxo_kpRudFiyVenyr4BsvoNH4NVoVbzJELnQ5g"

QUOTED_OR_PLAIN = re.compile(r'"[^"]*"|“[^”]*”|[^,]+')
QUOTE_CHARS = '"“”'
SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


# ============================================================================
# Category terms
# ============================================================================

def split_terms(cell: str) -> List[str]:
    """Split a sheet cell on commas outside quotes and strip the quotes."""
    terms = []
    for chunk in QUOTED_OR_PLAIN.findall(cell):
        term = chunk.strip().strip(QUOTE_CHARS).strip()
        if term:
            terms.append(term)
    return terms


def term_pattern(term: str) -> re.Pattern:
    """Whole-word, case-insensitive pattern for a word or phrase.

    Words in a phrase may be separated by any whitespace (including line
    breaks), and straight/curly apostrophes match each other.
    """
    words = [re.escape(w) for w in term.split()]
    body = r"\s+".join(words).replace("'", "['’]").replace("’", "['’]")
    return re.compile(r"\b" + body + r"\b", re.IGNORECASE)


def fetch_categories(sheet_id: str = DEFAULT_SHEET_ID) -> Dict[str, List[str]]:
    """Fetch the public Google Sheet and return {category_name: [terms]}."""
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    sheet = pd.read_csv(io.StringIO(resp.text))
    categories: Dict[str, List[str]] = {}

    for col in sheet.columns:
        seen: set = set()
        terms: List[str] = []
        for val in sheet[col].dropna():
            for term in split_terms(str(val)):
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

def sentencize(text: str) -> List[str]:
    """Split text into sentences using punctuation + capital letter boundaries."""
    if not text or not text.strip():
        return []
    return [s.strip() for s in SENT_BOUNDARY.split(text) if s.strip()]


# ============================================================================
# Google Sheet output
# ============================================================================

def open_spreadsheet(sheet_id: str, credentials_path: Optional[str]):
    """Authorise with a service account and open the output spreadsheet.

    Credentials come from the GOOGLE_CREDENTIALS env var (GitHub Actions) or
    a JSON key file.
    """
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds_json = os.environ.get("GOOGLE_CREDENTIALS")
    if creds_json:
        creds = Credentials.from_service_account_info(json.loads(creds_json), scopes=scopes)
    elif credentials_path:
        creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
    else:
        print("No credentials provided. Use --credentials or set GOOGLE_CREDENTIALS env var.",
              file=sys.stderr)
        sys.exit(1)

    return gspread.authorize(creds).open_by_key(sheet_id)


def write_tab(spreadsheet, tab_name: str, rows: List[list], freeze_rows: int = 1) -> None:
    """Clear (or create) a tab and write rows starting at A1."""
    import gspread

    # Sheets tab names are limited to 100 characters
    tab_name = tab_name[:100]
    n_cols = max((len(r) for r in rows), default=1)
    try:
        worksheet = spreadsheet.worksheet(tab_name)
        worksheet.clear()
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=tab_name, rows=len(rows) + 1, cols=n_cols)

    worksheet.update(range_name="A1", values=rows)
    if freeze_rows:
        worksheet.freeze(rows=freeze_rows)
    print(f"Wrote {len(rows)} rows to Google Sheet tab '{tab_name}'")
