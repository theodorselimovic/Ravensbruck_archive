"""Named Entity Recognition on Ravensbrück testimonies.

Segments testimony text into sentences, extracts named entities using a
multilingual BERT NER model (Davlan/bert-base-multilingual-cased-ner-hrl)
for PER/LOC/ORG and spaCy's English pipeline for DATE entities. Produces
entity-level, sentence-level, and document-level aggregations.
The multilingual model handles both English text and Polish proper nouns.

Usage:
    python 05_ner.py
    python 05_ner.py --verbose
    python 05_ner.py --device mps --batch-size 16
    python 05_ner.py --entity-types PER LOC ORG
    python 05_ner.py --max-docs 10 --verbose

Requirements:
    pip install transformers torch pandas pyarrow spacy
    python -m spacy download en_core_web_sm

Model:
    Davlan/bert-base-multilingual-cased-ner-hrl (Hugging Face)
    Multilingual BERT fine-tuned on 10 languages including English and Polish.
    Entity types: PER, LOC, ORG
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "metadata.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "ner"

MODEL_NAME = "Davlan/bert-base-multilingual-cased-ner-hrl"

ENTITY_TYPES = ["PER", "LOC", "ORG", "DATE"]

ENTITY_DESCRIPTIONS = {
    "PER": "Person names (survivors, perpetrators, family, witnesses)",
    "LOC": "Locations (camps, cities, geographic features)",
    "ORG": "Organizations (SS, Red Cross, institutions)",
    "DATE": "Date expressions (arrest dates, periods, years)",
}

# Entity types extracted by the multilingual BERT model
BERT_ENTITY_TYPES = {"PER", "LOC", "ORG"}

# Entity types extracted by spaCy (supplementary)
SPACY_ENTITY_TYPES = {"DATE"}

DEFAULT_BATCH_SIZES = {
    "cuda": 32,
    "mps": 16,
    "cpu": 8,
}

CHECKPOINT_FREQUENCY = 50

DEFAULT_MIN_CONFIDENCE = 0.5

# Minimum entity text length (filters subword fragments)
MIN_ENTITY_LENGTH = 2


# ============================================================================
# Logging
# ============================================================================

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
class Entity:
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float
    sentence_idx: int
    doc_id: str


@dataclass
class ProcessingStats:
    total_documents: int = 0
    total_sentences: int = 0
    sentences_processed: int = 0
    total_entities: int = 0
    entities_by_type: Dict[str, int] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    device_used: str = ""
    batch_size: int = 0
    errors: int = 0
    oom_recoveries: int = 0


# ============================================================================
# Sentencizer
# ============================================================================

class Sentencizer:
    """Splits document text into sentences using spaCy's rule-based sentencizer."""

    def __init__(self):
        import spacy
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")
        self.nlp.max_length = 500_000

    def sentencize(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


# ============================================================================
# Date Extractor (spaCy)
# ============================================================================

class DateExtractor:
    """Extracts DATE entities using spaCy's English NER pipeline.

    Supplements the multilingual BERT model which does not produce DATE labels.
    """

    def __init__(self):
        import spacy
        for model_name in ["en_core_web_sm", "en_core_web_lg", "en_core_web_trf"]:
            try:
                self.nlp = spacy.load(model_name, disable=["lemmatizer", "textcat"])
                self.nlp.max_length = 500_000
                logger.info(f"Date extractor loaded spaCy model: {model_name}")
                return
            except OSError:
                continue
        raise ImportError(
            "No spaCy English model found. Run:\n"
            "  python -m spacy download en_core_web_sm"
        )

    def extract_dates(
        self,
        sentences: List[Tuple[str, int, str]],
    ) -> List[Entity]:
        """Extract DATE entities from pre-sentencized text.

        Parameters
        ----------
        sentences : list of (sentence_text, sentence_idx, doc_id)

        Returns
        -------
        list of Entity
        """
        texts = [s[0] for s in sentences]
        entities = []

        for doc, (_, sent_idx, doc_id) in zip(self.nlp.pipe(texts, batch_size=100), sentences):
            for ent in doc.ents:
                if ent.label_ != "DATE":
                    continue
                if len(ent.text.strip()) < MIN_ENTITY_LENGTH:
                    continue
                entities.append(Entity(
                    text=ent.text.strip(),
                    entity_type="DATE",
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0,
                    sentence_idx=sent_idx,
                    doc_id=doc_id,
                ))

        return entities


# ============================================================================
# Device Manager
# ============================================================================

class DeviceManager:
    """Manages device detection and selection for PyTorch inference."""

    def __init__(self, requested_device: str = "auto"):
        import torch
        self.torch = torch
        self.requested_device = requested_device
        self.device = self._detect_device()
        self.device_type = self._get_device_type()

    def _detect_device(self) -> "torch.device":
        torch = self.torch

        if self.requested_device != "auto":
            if self.requested_device == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            elif (
                self.requested_device == "mps"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")
            elif self.requested_device == "cpu":
                return torch.device("cpu")
            else:
                logger.warning(
                    f"Requested device '{self.requested_device}' not available, "
                    "falling back to auto-detection"
                )

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _get_device_type(self) -> str:
        device_str = str(self.device)
        if "cuda" in device_str:
            return "cuda"
        elif "mps" in device_str:
            return "mps"
        else:
            return "cpu"

    def get_recommended_batch_size(self) -> int:
        return DEFAULT_BATCH_SIZES.get(self.device_type, 8)

    def get_device_info(self) -> Dict[str, str]:
        torch = self.torch
        info = {
            "device": str(self.device),
            "device_type": self.device_type,
            "torch_version": torch.__version__,
        }
        if self.device_type == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = (
                f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}"
            )
        elif self.device_type == "mps":
            info["gpu_name"] = "Apple Silicon (MPS)"
        return info


# ============================================================================
# NER Extractor
# ============================================================================

class NERExtractor:
    """Extracts named entities using HuggingFace transformer NER model."""

    def __init__(
        self,
        device_manager: DeviceManager,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        entity_types: Optional[List[str]] = None,
    ):
        self.device_manager = device_manager
        self.min_confidence = min_confidence
        self.entity_types = set(entity_types) if entity_types else set(ENTITY_TYPES)
        self.pipeline = None
        self._loaded = False

    def load_model(self) -> None:
        if self._loaded:
            return

        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

        logger.info(f"Loading model: {MODEL_NAME}")
        logger.info(f"Device: {self.device_manager.device}")

        start_time = time.time()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
        model.to(self.device_manager.device)
        model.eval()

        self.pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            device=self.device_manager.device,
            aggregation_strategy="simple",
        )

        elapsed = time.time() - start_time
        logger.info(f"Model loaded in {elapsed:.1f}s")
        self._loaded = True

    def extract_entities(
        self,
        text: str,
        sentence_idx: int,
        doc_id: str,
    ) -> List[Entity]:
        if not self._loaded:
            self.load_model()

        if not text or not text.strip():
            return []

        try:
            results = self.pipeline(text)
            entities = []
            for result in results:
                entity_group = result.get("entity_group", result.get("entity", ""))
                if "-" in entity_group:
                    entity_type = entity_group.split("-")[-1]
                else:
                    entity_type = entity_group

                if entity_type not in self.entity_types:
                    continue

                confidence = result.get("score", 0.0)
                if confidence < self.min_confidence:
                    continue

                entity_text = result.get("word", "").strip()

                # Skip subword fragments and single-char noise
                if "##" in entity_text or len(entity_text) < MIN_ENTITY_LENGTH:
                    continue

                entity = Entity(
                    text=entity_text,
                    entity_type=entity_type,
                    start=result.get("start", 0),
                    end=result.get("end", 0),
                    confidence=confidence,
                    sentence_idx=sentence_idx,
                    doc_id=doc_id,
                )
                entities.append(entity)

            return entities

        except Exception as e:
            logger.debug(f"Error processing text: {e}")
            return []

    def extract_batch(
        self,
        texts: List[str],
        sentence_indices: List[int],
        doc_ids: List[str],
    ) -> List[Entity]:
        if not self._loaded:
            self.load_model()

        all_entities = []

        try:
            valid_data = [
                (text, idx, doc_id)
                for text, idx, doc_id in zip(texts, sentence_indices, doc_ids)
                if text and text.strip()
            ]

            if not valid_data:
                return []

            valid_texts = [d[0] for d in valid_data]
            batch_results = self.pipeline(valid_texts)

            for i, results in enumerate(batch_results):
                _, sentence_idx, doc_id = valid_data[i]

                if isinstance(results, dict):
                    results = [results]

                for result in results:
                    entity_group = result.get(
                        "entity_group", result.get("entity", "")
                    )
                    if "-" in entity_group:
                        entity_type = entity_group.split("-")[-1]
                    else:
                        entity_type = entity_group

                    if entity_type not in self.entity_types:
                        continue

                    confidence = result.get("score", 0.0)
                    if confidence < self.min_confidence:
                        continue

                    entity_text = result.get("word", "").strip()

                    if "##" in entity_text or len(entity_text) < MIN_ENTITY_LENGTH:
                        continue

                    entity = Entity(
                        text=entity_text,
                        entity_type=entity_type,
                        start=result.get("start", 0),
                        end=result.get("end", 0),
                        confidence=confidence,
                        sentence_idx=sentence_idx,
                        doc_id=doc_id,
                    )
                    all_entities.append(entity)

        except Exception as e:
            logger.warning(f"Batch processing error: {e}")
            for text, idx, doc_id in zip(texts, sentence_indices, doc_ids):
                entities = self.extract_entities(text, idx, doc_id)
                all_entities.extend(entities)

        return all_entities


# ============================================================================
# NER Processor
# ============================================================================

class NERProcessor:
    """Orchestrates NER extraction: loading, sentencizing, processing, output."""

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        device: str = "auto",
        batch_size: Optional[int] = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        entity_types: Optional[List[str]] = None,
        max_docs: Optional[int] = None,
        verbose: bool = False,
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.max_docs = max_docs
        self.checkpoint_dir = output_dir / "checkpoints"
        self.verbose = verbose

        self.device_manager = DeviceManager(device)
        self.batch_size = batch_size or self.device_manager.get_recommended_batch_size()

        self.extractor = NERExtractor(
            device_manager=self.device_manager,
            min_confidence=min_confidence,
            entity_types=entity_types,
        )

        self.sentencizer = Sentencizer()
        self.date_extractor: Optional[DateExtractor] = None

        self.df_input: Optional[pd.DataFrame] = None
        self.sentences: List[Tuple[str, int, str]] = []
        self.entities: List[Entity] = []
        self.stats = ProcessingStats()

    def load_data(self) -> pd.DataFrame:
        logger.info("=" * 60)
        logger.info("LOADING INPUT DATA")
        logger.info("=" * 60)

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        df = pd.read_parquet(self.input_path)

        if "testimony_body" not in df.columns:
            raise ValueError(
                "Input must have 'testimony_body' column. "
                f"Available: {list(df.columns)}"
            )
        if "doc_id" not in df.columns:
            raise ValueError("Input must have 'doc_id' column")

        logger.info(f"Loaded {len(df):,} documents")

        if self.max_docs and len(df) > self.max_docs:
            logger.info(f"Limiting to {self.max_docs:,} documents (--max-docs)")
            df = df.head(self.max_docs).copy()

        self.df_input = df
        self.stats.total_documents = len(df)
        return df

    def sentencize_corpus(self) -> List[Tuple[str, int, str]]:
        """Segment all documents into sentences.

        Returns list of (sentence_text, sentence_idx, doc_id) tuples.
        """
        logger.info("=" * 60)
        logger.info("SENTENCIZING CORPUS")
        logger.info("=" * 60)

        sentences = []
        for _, row in self.df_input.iterrows():
            doc_id = str(row["doc_id"])
            text = row["testimony_body"]

            if not isinstance(text, str) or not text.strip():
                continue

            doc_sentences = self.sentencizer.sentencize(text)
            for idx, sent in enumerate(doc_sentences):
                sentences.append((sent, idx, doc_id))

        self.sentences = sentences
        self.stats.total_sentences = len(sentences)
        logger.info(
            f"Segmented {self.stats.total_documents:,} documents "
            f"into {len(sentences):,} sentences"
        )
        logger.info(
            f"Mean sentences/doc: "
            f"{len(sentences) / max(self.stats.total_documents, 1):.1f}"
        )
        return sentences

    def _check_resume(self) -> Tuple[int, List[Entity]]:
        checkpoint_file = self.checkpoint_dir / "ner_checkpoint.json"
        entities_file = self.checkpoint_dir / "ner_entities_partial.csv"

        if not checkpoint_file.exists() or not entities_file.exists():
            return 0, []

        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)

            processed = checkpoint.get("sentences_processed", 0)
            entities_df = pd.read_csv(entities_file)
            entities = [
                Entity(
                    text=row["text"],
                    entity_type=row["entity_type"],
                    start=row["start"],
                    end=row["end"],
                    confidence=row["confidence"],
                    sentence_idx=row["sentence_idx"],
                    doc_id=str(row["doc_id"]),
                )
                for _, row in entities_df.iterrows()
            ]

            logger.info(
                f"Resuming from checkpoint: {processed:,} sentences processed"
            )
            return processed, entities

        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            return 0, []

    def _save_checkpoint(self, processed: int, entities: List[Entity]) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = self.checkpoint_dir / "ner_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump({"sentences_processed": processed}, f)

        entities_file = self.checkpoint_dir / "ner_entities_partial.csv"
        entities_df = pd.DataFrame([asdict(e) for e in entities])
        entities_df.to_csv(entities_file, index=False)

        logger.debug(f"Checkpoint saved: {processed:,} sentences")

    def _clear_checkpoint(self) -> None:
        for f in [
            self.checkpoint_dir / "ner_checkpoint.json",
            self.checkpoint_dir / "ner_entities_partial.csv",
        ]:
            if f.exists():
                f.unlink()

    def process(self) -> List[Entity]:
        logger.info("=" * 60)
        logger.info("NER EXTRACTION")
        logger.info("=" * 60)

        device_info = self.device_manager.get_device_info()
        logger.info(
            f"Device: {device_info['device']} "
            f"({device_info.get('gpu_name', 'CPU')})"
        )
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Entity types: {sorted(self.extractor.entity_types)}")
        logger.info(f"Min confidence: {self.extractor.min_confidence}")

        self.extractor.load_model()

        start_idx, self.entities = self._check_resume()

        texts = [s[0] for s in self.sentences]
        sentence_indices = [s[1] for s in self.sentences]
        doc_ids = [s[2] for s in self.sentences]

        start_time = time.time()
        total = len(texts)
        current_batch_size = self.batch_size

        for i in range(start_idx, total, current_batch_size):
            batch_end = min(i + current_batch_size, total)

            batch_texts = texts[i:batch_end]
            batch_indices = sentence_indices[i:batch_end]
            batch_docs = doc_ids[i:batch_end]

            try:
                batch_entities = self.extractor.extract_batch(
                    batch_texts, batch_indices, batch_docs
                )
                self.entities.extend(batch_entities)
                self.stats.sentences_processed = batch_end

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.stats.oom_recoveries += 1
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.warning(
                        f"OOM error - reducing batch size to {current_batch_size}"
                    )
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    self.stats.errors += 1
                    logger.error(f"Error processing batch {i}: {e}")
                    continue

            except Exception as e:
                self.stats.errors += 1
                logger.error(f"Error processing batch {i}: {e}")
                continue

            if (batch_end % 1000 == 0) or batch_end == total:
                elapsed = time.time() - start_time
                rate = batch_end / elapsed if elapsed > 0 else 0
                eta = (total - batch_end) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {batch_end:,}/{total:,} sentences "
                    f"({batch_end / total:.1%}) | "
                    f"{len(self.entities):,} entities | "
                    f"ETA: {eta / 60:.1f} min"
                )

            sentences_so_far = batch_end
            docs_so_far = len(set(doc_ids[:batch_end]))
            if docs_so_far % CHECKPOINT_FREQUENCY == 0 and docs_so_far > 0:
                self._save_checkpoint(sentences_so_far, self.entities)

        # Date extraction via spaCy (BERT model doesn't produce DATE)
        if "DATE" in self.extractor.entity_types:
            logger.info("\nExtracting DATE entities via spaCy...")
            date_start = time.time()
            try:
                self.date_extractor = DateExtractor()
                date_entities = self.date_extractor.extract_dates(self.sentences)
                self.entities.extend(date_entities)
                logger.info(
                    f"Extracted {len(date_entities):,} DATE entities "
                    f"in {time.time() - date_start:.1f}s"
                )
            except ImportError as e:
                logger.warning(f"Skipping DATE extraction: {e}")

        elapsed = time.time() - start_time
        self.stats.processing_time_seconds = elapsed
        self.stats.total_entities = len(self.entities)
        self.stats.device_used = device_info["device"]
        self.stats.batch_size = self.batch_size

        for entity in self.entities:
            self.stats.entities_by_type[entity.entity_type] = (
                self.stats.entities_by_type.get(entity.entity_type, 0) + 1
            )

        self._clear_checkpoint()

        logger.info(f"\nExtraction complete in {elapsed:.1f}s")
        logger.info(f"Total entities: {len(self.entities):,}")

        return self.entities

    def create_entity_dataframe(self) -> pd.DataFrame:
        if not self.entities:
            return pd.DataFrame(
                columns=[
                    "text", "entity_type", "start", "end",
                    "confidence", "sentence_idx", "doc_id",
                ]
            )
        return pd.DataFrame([asdict(e) for e in self.entities])

    def aggregate_by_sentence(self) -> pd.DataFrame:
        if not self.entities:
            return pd.DataFrame()

        entity_df = self.create_entity_dataframe()

        counts = (
            entity_df
            .groupby(["doc_id", "sentence_idx", "entity_type"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        for etype in ENTITY_TYPES:
            if etype not in counts.columns:
                counts[etype] = 0

        counts["total_entities"] = counts[ENTITY_TYPES].sum(axis=1)

        sentence_df = pd.DataFrame(
            self.sentences, columns=["sentence_text", "sentence_idx", "doc_id"]
        )
        sentence_df["doc_id"] = sentence_df["doc_id"].astype(str)
        counts["doc_id"] = counts["doc_id"].astype(str)

        counts = counts.merge(
            sentence_df,
            on=["doc_id", "sentence_idx"],
            how="right",
        ).fillna(0)

        for etype in ENTITY_TYPES:
            counts[etype] = counts[etype].astype(int)
        counts["total_entities"] = counts.get("total_entities", 0).astype(int)

        return counts

    def aggregate_by_document(self) -> pd.DataFrame:
        if not self.entities:
            return pd.DataFrame()

        entity_df = self.create_entity_dataframe()

        counts = (
            entity_df
            .groupby(["doc_id", "entity_type"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        for etype in ENTITY_TYPES:
            if etype not in counts.columns:
                counts[etype] = 0

        counts["total_entities"] = counts[ENTITY_TYPES].sum(axis=1)

        unique_counts = (
            entity_df
            .groupby(["doc_id", "entity_type"])["text"]
            .nunique()
            .unstack(fill_value=0)
            .reset_index()
        )
        unique_counts.columns = ["doc_id"] + [
            f"{c}_unique" for c in unique_counts.columns[1:]
        ]

        counts = counts.merge(unique_counts, on="doc_id", how="left")

        sentence_counts = (
            pd.DataFrame(self.sentences, columns=["text", "idx", "doc_id"])
            .groupby("doc_id")
            .size()
            .reset_index(name="sentence_count")
        )
        sentence_counts["doc_id"] = sentence_counts["doc_id"].astype(str)
        counts["doc_id"] = counts["doc_id"].astype(str)
        counts = counts.merge(sentence_counts, on="doc_id", how="left")

        if self.df_input is not None:
            meta_cols = ["doc_id"] + [
                c for c in ["name", "testimony_place", "testimony_date"]
                if c in self.df_input.columns
            ]
            doc_meta = self.df_input[meta_cols].copy()
            doc_meta["doc_id"] = doc_meta["doc_id"].astype(str)
            counts = counts.merge(doc_meta, on="doc_id", how="left")

        return counts

    def generate_report(self) -> Dict:
        entity_df = self.create_entity_dataframe()

        top_entities = {}
        for etype in ENTITY_TYPES:
            type_entities = entity_df[entity_df["entity_type"] == etype]
            if len(type_entities) > 0:
                top = type_entities["text"].value_counts().head(20).to_dict()
                top_entities[etype] = top

        return {
            "metadata": {
                "created": datetime.now().isoformat(),
                "input_file": str(self.input_path),
                "output_dir": str(self.output_dir),
                "model": MODEL_NAME,
                "device": self.stats.device_used,
                "batch_size": self.stats.batch_size,
                "min_confidence": self.extractor.min_confidence,
                "entity_types_requested": sorted(self.extractor.entity_types),
            },
            "statistics": {
                "total_documents": self.stats.total_documents,
                "total_sentences": self.stats.total_sentences,
                "sentences_processed": self.stats.sentences_processed,
                "total_entities": self.stats.total_entities,
                "entities_by_type": self.stats.entities_by_type,
                "unique_entities_by_type": {
                    etype: int(
                        entity_df[entity_df["entity_type"] == etype]["text"].nunique()
                    )
                    for etype in ENTITY_TYPES
                    if etype in entity_df["entity_type"].values
                },
                "processing_time_seconds": round(
                    self.stats.processing_time_seconds, 1
                ),
                "sentences_per_second": round(
                    self.stats.sentences_processed
                    / max(self.stats.processing_time_seconds, 0.1),
                    1,
                ),
                "errors": self.stats.errors,
                "oom_recoveries": self.stats.oom_recoveries,
            },
            "top_entities": top_entities,
            "entity_type_descriptions": ENTITY_DESCRIPTIONS,
        }

    def save_outputs(self) -> None:
        logger.info("=" * 60)
        logger.info("SAVING OUTPUTS")
        logger.info("=" * 60)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        entity_df = self.create_entity_dataframe()
        entities_path = self.output_dir / "entities.csv"
        entity_df.to_csv(entities_path, index=False)
        logger.info(f"Saved: {entities_path} ({len(entity_df):,} entities)")

        sentence_df = self.aggregate_by_sentence()
        sentence_path = self.output_dir / "entities_by_sentence.csv"
        sentence_df.to_csv(sentence_path, index=False)
        logger.info(f"Saved: {sentence_path} ({len(sentence_df):,} rows)")

        doc_df = self.aggregate_by_document()
        doc_path = self.output_dir / "entities_by_document.csv"
        doc_df.to_csv(doc_path, index=False)
        logger.info(f"Saved: {doc_path} ({len(doc_df):,} documents)")

        report = self.generate_report()
        report_path = self.output_dir / "ner_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved: {report_path}")

    def print_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 60)

        logger.info(f"\nDocuments processed: {self.stats.total_documents:,}")
        logger.info(f"Sentences processed: {self.stats.sentences_processed:,}")
        logger.info(f"Total entities extracted: {self.stats.total_entities:,}")
        logger.info(
            f"Processing time: {self.stats.processing_time_seconds:.1f}s"
        )

        logger.info("\nEntities by type:")
        for etype in ENTITY_TYPES:
            count = self.stats.entities_by_type.get(etype, 0)
            desc = ENTITY_DESCRIPTIONS.get(etype, "")
            logger.info(f"  {etype}: {count:,} ({desc})")

        if self.stats.errors > 0:
            logger.info(f"\nErrors encountered: {self.stats.errors}")
        if self.stats.oom_recoveries > 0:
            logger.info(f"OOM recoveries: {self.stats.oom_recoveries}")

        if self.entities:
            logger.info("\nExample entities:")
            for etype in ENTITY_TYPES:
                type_entities = [
                    e for e in self.entities if e.entity_type == etype
                ]
                if type_entities:
                    sample = type_entities[:3]
                    examples = ", ".join(f'"{e.text}"' for e in sample)
                    logger.info(f"  {etype}: {examples}")


# ============================================================================
# Pipeline
# ============================================================================

def run_pipeline(
    input_path: Path,
    output_dir: Path,
    device: str = "auto",
    batch_size: Optional[int] = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    entity_types: Optional[List[str]] = None,
    max_docs: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    setup_logging(verbose)

    logger.info("=" * 60)
    logger.info("NER EXTRACTION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Model:  {MODEL_NAME}")
    logger.info("")

    processor = NERProcessor(
        input_path=input_path,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        min_confidence=min_confidence,
        entity_types=entity_types,
        max_docs=max_docs,
        verbose=verbose,
    )

    processor.load_data()
    processor.sentencize_corpus()
    processor.process()
    processor.save_outputs()
    processor.print_summary()

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("\nFiles generated:")
    logger.info("  - entities.csv: All extracted entities")
    logger.info("  - entities_by_sentence.csv: Entity counts per sentence")
    logger.info("  - entities_by_document.csv: Entity counts per document")
    logger.info("  - ner_report.json: Summary statistics")

    return processor.generate_report()


# ============================================================================
# CLI
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract named entities from Ravensbrück testimonies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Entity Types:
    PER:  Person names (survivors, perpetrators, family, witnesses)
    LOC:  Locations (camps, cities, geographic features)
    ORG:  Organizations (SS, Red Cross, institutions)

Examples:
    # Basic usage (auto-detect device)
    python 05_ner.py

    # Run on Apple Silicon with MPS
    python 05_ner.py --device mps --batch-size 16

    # Extract only persons and locations
    python 05_ner.py --entity-types PER LOC

    # Test on small sample
    python 05_ner.py --max-docs 10 --verbose

Model:
    Davlan/bert-base-multilingual-cased-ner-hrl (Hugging Face)
    https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input parquet file (default: {DEFAULT_INPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device for inference (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for inference (default: auto based on device)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"Minimum confidence score 0.0-1.0 (default: {DEFAULT_MIN_CONFIDENCE})",
    )
    parser.add_argument(
        "--entity-types",
        type=str,
        nargs="+",
        choices=ENTITY_TYPES,
        default=None,
        help=f"Entity types to extract (default: all). Options: {', '.join(ENTITY_TYPES)}",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to process (for testing)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def main() -> int:
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        run_pipeline(
            input_path=args.input,
            output_dir=args.output,
            device=args.device,
            batch_size=args.batch_size,
            min_confidence=args.min_confidence,
            entity_types=args.entity_types,
            max_docs=args.max_docs,
            verbose=args.verbose,
        )
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except ImportError as e:
        print(f"Error: Missing dependency - {e}", file=sys.stderr)
        print("\nInstall required packages:", file=sys.stderr)
        print("  pip install transformers torch pandas pyarrow", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        logging.exception("Unexpected error during processing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
