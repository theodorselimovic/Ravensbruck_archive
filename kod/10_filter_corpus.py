"""Filter the corpus for sentences containing a specific word.

Searches testimony text at the sentence level and exports matching rows
with document metadata. Searches the original surface text (not lemmatized
BoW tokens), so the query should match the word as it appears in text.
Case-insensitive by default.

Usage:
    python 10_filter_corpus.py hunger
    python 10_filter_corpus.py "Red Cross"
    python 10_filter_corpus.py beaten --case-sensitive
    python 10_filter_corpus.py hunger --gender female

Requirements:
    pandas, pyarrow, spacy
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

try:
    import spacy
except ImportError:
    print(
        "spaCy is required but not installed. Run:\n"
        "  pip install spacy",
        file=sys.stderr,
    )
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "metadata.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "filtered"


# ============================================================================
# Sentencizer
# ============================================================================

class Sentencizer:
    """Splits document text into sentences using spaCy's rule-based sentencizer."""

    def __init__(self):
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")
        self.nlp.max_length = 500_000

    def sentencize(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter corpus for sentences containing a specific word or phrase.",
    )
    parser.add_argument(
        "query",
        help="Word or phrase to search for (e.g. 'hunger', 'Red Cross')",
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
        "--case-sensitive",
        action="store_true",
        help="Match case-sensitively (default: case-insensitive)",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default=None,
        choices=["female", "male"],
        help="Filter corpus by gender before searching",
    )
    args = parser.parse_args()

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

    flags = 0 if args.case_sensitive else re.IGNORECASE
    pattern = re.compile(r"\b" + re.escape(args.query) + r"\b", flags)

    sentencizer = Sentencizer()
    rows: List[dict] = []

    for _, doc_row in df.iterrows():
        raw_id = doc_row["testimony_number"]
        if pd.isna(raw_id):
            doc_id = "unknown"
        elif isinstance(raw_id, float) and raw_id == int(raw_id):
            doc_id = str(int(raw_id))
        else:
            doc_id = str(raw_id)

        gender = doc_row.get("gender", "")
        text = doc_row[text_col]
        if not text or not isinstance(text, str):
            continue

        sentences = sentencizer.sentencize(text)
        for sent_idx, sent in enumerate(sentences):
            if pattern.search(sent):
                rows.append({
                    "doc_id": doc_id,
                    "sentence_idx": sent_idx,
                    "sentence": sent,
                    "gender": gender,
                })

    if not rows:
        print(f"No sentences found containing '{args.query}'.")
        sys.exit(0)

    result = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w\-]", "_", args.query.lower())
    parts = [safe_name]
    if args.gender:
        parts.append(args.gender)
    filename = "_".join(parts) + ".csv"
    output_path = args.output_dir / filename

    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} sentences to {output_path}")


if __name__ == "__main__":
    main()
