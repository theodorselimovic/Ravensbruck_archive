"""Filter verb records for a specific verb lemma.

Reads verbs.parquet and exports matching rows to a CSV file,
making it easy to inspect sentence contexts for a given verb.

Usage:
    python 09_filter_verbs.py beat
    python 09_filter_verbs.py beat --voice passive
    python 09_filter_verbs.py take --polarity negated
    python 09_filter_verbs.py kill --voice passive --polarity negated
    python 09_filter_verbs.py beat -i data/verbs_female/verbs.parquet
    python 09_filter_verbs.py --list

Requirements:
    pandas, pyarrow
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "verbs" / "verbs.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "verbs" / "filtered"


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter verb records by lemma, voice, and/or polarity.",
    )
    parser.add_argument(
        "verb",
        nargs="?",
        help="Verb lemma to filter for (e.g. 'beat', 'take', 'kill')",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input verbs parquet file (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for CSV files (default: %(default)s)",
    )
    parser.add_argument(
        "--voice",
        choices=["active", "passive"],
        default=None,
        help="Filter by voice (default: both)",
    )
    parser.add_argument(
        "--polarity",
        choices=["affirmative", "negated"],
        default=None,
        help="Filter by polarity (default: both)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available verb lemmas with counts and exit",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        print("Run 08_verb_analysis.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(args.input)

    if args.list:
        counts = df["verb_lemma"].value_counts().head(50)
        print(f"Top 50 verb lemmas ({len(df)} total records, "
              f"{df['verb_lemma'].nunique()} unique lemmas):\n")
        for lemma, count in counts.items():
            print(f"  {lemma:20s} {count:>6d}")
        return

    if not args.verb:
        parser.error("verb lemma is required (or use --list)")

    mask = df["verb_lemma"] == args.verb
    if not mask.any():
        print(f"No records found for verb lemma '{args.verb}'.", file=sys.stderr)
        close = df["verb_lemma"].value_counts()
        suggestions = [l for l in close.index if l.startswith(args.verb[:3])][:5]
        if suggestions:
            print(f"Did you mean: {', '.join(suggestions)}?", file=sys.stderr)
        sys.exit(1)

    if args.voice:
        mask &= df["voice"] == args.voice
    if args.polarity:
        mask &= df["polarity"] == args.polarity

    result = df[mask]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts = [args.verb]
    if args.voice:
        parts.append(args.voice)
    if args.polarity:
        parts.append(args.polarity)
    filename = "_".join(parts) + ".csv"
    output_path = args.output_dir / filename

    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} rows to {output_path}")


if __name__ == "__main__":
    main()
