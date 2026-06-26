"""BoW preprocessing: stopword removal and lemmatization.

Reads testimony text from metadata.parquet and produces a lemmatized,
stopword-free token list for each document using spaCy's English pipeline.

Usage:
    python 03_preprocess_bow.py
    python 03_preprocess_bow.py --verbose
    python 03_preprocess_bow.py --input data/corpus.parquet --text-column full_text
    python 03_preprocess_bow.py --keep-stopwords
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

try:
    import spacy
except ImportError:
    print(
        "spaCy is required but not installed. Run:\n"
        "  pip install spacy && python -m spacy download en_core_web_sm",
        file=sys.stderr,
    )
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "metadata.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "bow_corpus.parquet"

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================================
# Text Processing
# ============================================================================

class BoWPreprocessor:
    """Preprocessor for bag-of-words analysis using spaCy lemmatization."""

    def __init__(
        self,
        custom_stopwords: Optional[Set[str]] = None,
        keep_stopwords: bool = False,
        min_token_length: int = 2,
    ):
        self.keep_stopwords = keep_stopwords
        self.min_token_length = min_token_length

        self.nlp = self._load_spacy_model()

        if custom_stopwords:
            for word in custom_stopwords:
                self.nlp.vocab[word].is_stop = True

    def _load_spacy_model(self) -> spacy.language.Language:
        try:
            nlp = spacy.load(
                "en_core_web_sm",
                disable=["parser", "ner"],
            )
        except OSError:
            print(
                "spaCy model 'en_core_web_sm' not found. Run:\n"
                "  python -m spacy download en_core_web_sm",
                file=sys.stderr,
            )
            sys.exit(1)

        nlp.max_length = 300_000
        return nlp

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        return text.strip()

    def _lemmatize_and_filter(self, doc: spacy.tokens.Doc) -> List[str]:
        tokens = []
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            if token.like_num or token.text.isdigit():
                continue
            if len(token.text) < self.min_token_length:
                continue
            if not self.keep_stopwords and token.is_stop:
                continue

            lemma = token.lemma_.lower()

            if len(lemma) < self.min_token_length:
                continue
            if not any(c.isalpha() for c in lemma):
                continue

            tokens.append(lemma)

        return tokens

    def process_document(self, text: str) -> List[str]:
        text = self._clean_text(text)
        if not text:
            return []
        doc = self.nlp(text)
        return self._lemmatize_and_filter(doc)

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "testimony_body",
    ) -> pd.DataFrame:
        total = len(df)
        logger.info(f"Processing {total:,} documents...")

        texts = [self._clean_text(t) for t in df[text_column]]
        tokens_list = []
        start_time = time.time()

        for i, doc in enumerate(self.nlp.pipe(texts, batch_size=50)):
            tokens = self._lemmatize_and_filter(doc)
            tokens_list.append(tokens)

            if (i + 1) % 50 == 0 or (i + 1) == total:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (total - i - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  {i + 1:,}/{total:,} "
                    f"({(i + 1) / total * 100:.0f}%) - "
                    f"{rate:.1f} docs/sec - "
                    f"ETA: {remaining:.0f}s"
                )

        df = df.copy()
        df["tokens"] = tokens_list
        df["tokens_text"] = df["tokens"].apply(lambda x: " ".join(x))
        df["token_count"] = df["tokens"].apply(len)

        elapsed = time.time() - start_time
        logger.info(f"Processing complete in {elapsed:.1f}s")
        return df


# ============================================================================
# Stopwords
# ============================================================================

def load_stopwords(filepath: Path) -> Set[str]:
    """Load stopwords from file (one per line, # comments allowed)."""
    stopwords = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word and not word.startswith("#"):
                stopwords.add(word)
    return stopwords


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BoW preprocessing: stopword removal and lemmatization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=DEFAULT_INPUT,
        help=f"Input parquet file (default: {DEFAULT_INPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output parquet file (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--text-column", type=str, default="testimony_body",
        help="Column containing text to process (default: testimony_body)",
    )
    parser.add_argument(
        "--keep-stopwords", action="store_true",
        help="Keep stopwords in output",
    )
    parser.add_argument(
        "--stopwords-file", type=Path, default=None,
        help="File with additional stopwords (one per line)",
    )
    parser.add_argument(
        "--min-token-length", type=int, default=2,
        help="Minimum token length to keep (default: 2)",
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
    logger.info("BOW PREPROCESSING (LEMMATIZATION)")
    logger.info("=" * 60)
    logger.info(f"Input:            {args.input}")
    logger.info(f"Output:           {args.output}")
    logger.info(f"Text column:      {args.text_column}")
    logger.info(f"Keep stopwords:   {args.keep_stopwords}")
    logger.info(f"Min token length: {args.min_token_length}")

    # ---- Load ----
    logger.info("\nLoading data...")
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} documents")

    if args.text_column not in df.columns:
        logger.error(
            f"Column '{args.text_column}' not found. "
            f"Available: {list(df.columns)}"
        )
        sys.exit(1)

    # ---- Custom stopwords ----
    custom_stopwords = None
    if args.stopwords_file:
        if args.stopwords_file.exists():
            custom_stopwords = load_stopwords(args.stopwords_file)
            logger.info(f"Loaded {len(custom_stopwords)} custom stopwords")
        else:
            logger.warning(f"Stopwords file not found: {args.stopwords_file}")

    # ---- Process ----
    logger.info("\n" + "=" * 60)
    logger.info("LEMMATIZING")
    logger.info("=" * 60)

    preprocessor = BoWPreprocessor(
        custom_stopwords=custom_stopwords,
        keep_stopwords=args.keep_stopwords,
        min_token_length=args.min_token_length,
    )

    df = preprocessor.process_dataframe(df, text_column=args.text_column)

    # ---- Summary ----
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    total_tokens = df["token_count"].sum()
    mean_tokens = df["token_count"].mean()
    median_tokens = df["token_count"].median()
    empty_docs = (df["token_count"] == 0).sum()

    all_lemmas = {t for tokens in df["tokens"] for t in tokens}

    logger.info(f"Total tokens:     {total_tokens:,}")
    logger.info(f"Vocabulary size:  {len(all_lemmas):,} unique lemmas")
    logger.info(f"Mean tokens/doc:  {mean_tokens:.0f}")
    logger.info(f"Median tokens/doc:{median_tokens:.0f}")
    logger.info(f"Empty documents:  {empty_docs}")

    # Sample output
    logger.info("\nSample output:")
    sample = df[df["token_count"] > 0].head(3)
    for _, row in sample.iterrows():
        original = row[args.text_column]
        snippet = original[:100] if isinstance(original, str) else ""
        logger.info(f"  Text:   {snippet}...")
        logger.info(f"  Tokens: {row['tokens'][:10]}")
        logger.info(f"  Count:  {row['token_count']}")
        logger.info("")

    # ---- Save ----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    file_size = args.output.stat().st_size / 1024 / 1024
    logger.info(f"Saved to {args.output} ({file_size:.1f} MB)")

    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
