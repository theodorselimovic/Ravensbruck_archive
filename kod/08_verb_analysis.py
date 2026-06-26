"""Verb extraction and voice/negation classification for Ravensbrück testimonies.

Segments testimony text into sentences, extracts all verbs using spaCy
dependency parsing, and classifies each along two axes:
  - voice: active / passive
  - polarity: affirmative / negated
Preserves full sentence context for each verb occurrence.

Usage:
    python 08_verb_analysis.py
    python 08_verb_analysis.py --verbose
    python 08_verb_analysis.py --max-docs 10 --verbose
    python 08_verb_analysis.py --gender female
    python 08_verb_analysis.py --gender female -o data/verbs_female
    python 08_verb_analysis.py --model en_core_web_trf --batch-size 64

Requirements:
    pip install spacy pandas pyarrow
    python -m spacy download en_core_web_sm
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import spacy
    from spacy.tokens import Token
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "verbs"
DEFAULT_MODEL = "en_core_web_sm"
DEFAULT_BATCH_SIZE = 256

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class VerbRecord:
    doc_id: str
    sentence_idx: int
    sentence_text: str
    verb_text: str
    verb_lemma: str
    verb_pos: str
    voice: str
    polarity: str
    verb_idx_in_sentence: int


@dataclass
class ProcessingStats:
    total_documents: int = 0
    total_sentences: int = 0
    total_verbs: int = 0
    active_count: int = 0
    passive_count: int = 0
    affirmative_count: int = 0
    negated_count: int = 0
    processing_time_seconds: float = 0.0
    model_name: str = ""
    gender_filter: str = ""
    errors: int = 0


# ============================================================================
# Voice Classifier
# ============================================================================

class VerbClassifier:
    """Classifies verb voice and polarity using spaCy dependency labels."""

    @staticmethod
    def classify_voice(token: Token) -> str:
        children_deps = {child.dep_ for child in token.children}
        if "nsubjpass" in children_deps or "auxpass" in children_deps:
            return "passive"
        return "active"

    @staticmethod
    def classify_polarity(token: Token) -> str:
        """Detect negation via the 'neg' dependency label (not, never, n't)."""
        for child in token.children:
            if child.dep_ == "neg":
                return "negated"
        return "affirmative"

    @staticmethod
    def is_verb(token: Token) -> bool:
        return token.pos_ == "VERB"


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
# Verb Analyser
# ============================================================================

class VerbAnalyser:
    """Extracts verbs and classifies voice across the testimony corpus."""

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_docs: Optional[int] = None,
        gender: Optional[str] = None,
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_docs = max_docs
        self.gender = gender

        self.sentencizer = Sentencizer()
        self.nlp = self._load_spacy_model()
        self.stats = ProcessingStats(model_name=model_name, gender_filter=gender or "")

    def _load_spacy_model(self) -> spacy.language.Language:
        try:
            nlp = spacy.load(self.model_name, disable=["ner"])
            nlp.max_length = 500_000
            logger.info("Loaded spaCy model: %s", self.model_name)
            return nlp
        except OSError:
            logger.error(
                "Model '%s' not found. Run:\n  python -m spacy download %s",
                self.model_name,
                self.model_name,
            )
            sys.exit(1)

    def load_data(self) -> pd.DataFrame:
        logger.info("Loading data from %s", self.input_path)
        df = pd.read_parquet(self.input_path)

        if "testimony" not in df.columns:
            candidates = [c for c in df.columns if c in ("testimony_body",) or "text" in c.lower() or "full" in c.lower()]
            if candidates:
                text_col = candidates[0]
                logger.info("Using column '%s' as text source", text_col)
                df = df.rename(columns={text_col: "testimony"})
            else:
                logger.error("No testimony/text column found. Columns: %s", list(df.columns))
                sys.exit(1)

        if "testimony_number" not in df.columns:
            df["testimony_number"] = df.index.astype(str)
            logger.info("No testimony_number column; using row index as doc_id")

        df = df.dropna(subset=["testimony"])
        df = df[df["testimony"].str.strip().astype(bool)]
        logger.info("Loaded %d documents with non-empty text", len(df))

        if self.gender:
            if "gender" not in df.columns:
                logger.error("No gender column found; cannot filter by gender")
                sys.exit(1)
            df = df[df["gender"].str.lower() == self.gender.lower()]
            logger.info("Filtered to %d %s documents", len(df), self.gender)

        if self.max_docs:
            df = df.head(self.max_docs)
            logger.info("Limited to %d documents (--max-docs)", self.max_docs)

        return df

    def process(self, df: pd.DataFrame) -> List[VerbRecord]:
        self.stats.total_documents = len(df)
        all_records: List[VerbRecord] = []
        start = time.time()

        for i, (_, row) in enumerate(df.iterrows()):
            raw_id = row["testimony_number"]
            if pd.isna(raw_id):
                doc_id = f"row_{i}"
            elif isinstance(raw_id, float) and raw_id == int(raw_id):
                doc_id = str(int(raw_id))
            else:
                doc_id = str(raw_id)
            text = row["testimony"]

            try:
                records = self._process_document(doc_id, text)
                all_records.extend(records)
            except Exception as e:
                logger.warning("Error processing document %s: %s", doc_id, e)
                self.stats.errors += 1
                continue

            if (i + 1) % 50 == 0 or (i + 1) == len(df):
                logger.info(
                    "Processed %d/%d documents (%d verbs so far)",
                    i + 1, len(df), len(all_records),
                )

        self.stats.processing_time_seconds = time.time() - start
        self.stats.total_verbs = len(all_records)
        self.stats.active_count = sum(1 for r in all_records if r.voice == "active")
        self.stats.passive_count = sum(1 for r in all_records if r.voice == "passive")
        self.stats.affirmative_count = sum(1 for r in all_records if r.polarity == "affirmative")
        self.stats.negated_count = sum(1 for r in all_records if r.polarity == "negated")

        return all_records

    def _process_document(self, doc_id: str, text: str) -> List[VerbRecord]:
        sentences = self.sentencizer.sentencize(text)
        self.stats.total_sentences += len(sentences)
        records: List[VerbRecord] = []

        parsed_sents = list(self.nlp.pipe(sentences, batch_size=self.batch_size))

        for sent_idx, (parsed, sent_text) in enumerate(zip(parsed_sents, sentences)):
            for token in parsed:
                if VerbClassifier.is_verb(token):
                    records.append(VerbRecord(
                        doc_id=doc_id,
                        sentence_idx=sent_idx,
                        sentence_text=sent_text,
                        verb_text=token.text,
                        verb_lemma=token.lemma_,
                        verb_pos=token.tag_,
                        voice=VerbClassifier.classify_voice(token),
                        polarity=VerbClassifier.classify_polarity(token),
                        verb_idx_in_sentence=token.i,
                    ))

        logger.debug(
            "Document %s: %d sentences, %d verbs",
            doc_id, len(sentences), len(records),
        )
        return records

    def save_outputs(self, records: List[VerbRecord]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # -- Verb-level parquet --
        df = pd.DataFrame([asdict(r) for r in records])
        verbs_path = self.output_dir / "verbs.parquet"
        df.to_parquet(verbs_path, index=False)
        logger.info("Saved %d verb records to %s", len(df), verbs_path)

        # -- Summary CSV --
        if not df.empty:
            voice_counts = df.groupby(["verb_lemma", "voice"]).size().unstack(fill_value=0)
            polarity_counts = df.groupby(["verb_lemma", "polarity"]).size().unstack(fill_value=0)
            summary = voice_counts.join(polarity_counts, how="outer").fillna(0).astype(int)
            for col in ["active", "passive", "affirmative", "negated"]:
                if col not in summary.columns:
                    summary[col] = 0
            summary["total"] = summary["active"] + summary["passive"]
            summary = summary[["active", "passive", "affirmative", "negated", "total"]]
            summary = summary.reset_index().sort_values("total", ascending=False)

            summary_path = self.output_dir / "verb_summary.csv"
            summary.to_csv(summary_path, index=False)
            logger.info("Saved verb summary to %s", summary_path)

        # -- Report JSON --
        report = asdict(self.stats)
        n = self.stats.total_verbs
        if n > 0:
            report["passive_ratio"] = self.stats.passive_count / n
            report["negated_ratio"] = self.stats.negated_count / n
        else:
            report["passive_ratio"] = 0.0
            report["negated_ratio"] = 0.0

        if not df.empty:
            for quadrant in ["active_affirmative", "active_negated",
                             "passive_affirmative", "passive_negated"]:
                voice, polarity = quadrant.split("_")
                mask = (df["voice"] == voice) & (df["polarity"] == polarity)
                report[f"{quadrant}_count"] = int(mask.sum())
                report[f"top_{quadrant}_verbs"] = (
                    df[mask].groupby("verb_lemma").size().nlargest(20).to_dict()
                )

        report_path = self.output_dir / "verb_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved processing report to %s", report_path)

    def log_summary(self) -> None:
        n = self.stats.total_verbs
        passive_ratio = self.stats.passive_count / n if n > 0 else 0.0
        negated_ratio = self.stats.negated_count / n if n > 0 else 0.0

        logger.info(
            "Done: %d documents, %d sentences, %d verbs",
            self.stats.total_documents,
            self.stats.total_sentences,
            n,
        )
        logger.info(
            "Voice:    active=%d, passive=%d (%.1f%% passive)",
            self.stats.active_count, self.stats.passive_count, passive_ratio * 100,
        )
        logger.info(
            "Polarity: affirmative=%d, negated=%d (%.1f%% negated)",
            self.stats.affirmative_count, self.stats.negated_count, negated_ratio * 100,
        )
        logger.info("Processing time: %.1fs", self.stats.processing_time_seconds)

        if passive_ratio < 0.05 or passive_ratio > 0.50:
            logger.warning(
                "Passive ratio %.1f%% is outside expected range (5-50%%). "
                "Check model or data.",
                passive_ratio * 100,
            )


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract verbs and classify voice from Ravensbrück testimonies.",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input parquet file (default: %(default)s)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="spaCy model name (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for nlp.pipe (default: %(default)s)",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default=None,
        choices=["female", "male"],
        help="Filter corpus by gender (default: all)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit number of documents (for testing)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    analyser = VerbAnalyser(
        input_path=args.input,
        output_dir=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
        gender=args.gender,
    )

    df = analyser.load_data()
    records = analyser.process(df)
    analyser.save_outputs(records)
    analyser.log_summary()


if __name__ == "__main__":
    main()
