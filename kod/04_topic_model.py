"""LDA topic modeling on lemmatized testimony corpus.

Trains Latent Dirichlet Allocation models over a range of topic counts,
evaluates coherence, and saves the best model with document-topic distributions.

Usage:
    python 04_topic_model.py                              # explore k=5,10,15,20,25
    python 04_topic_model.py --num-topics 15              # train single model with k=15
    python 04_topic_model.py --num-topics 5 10 15 20 25 30 --passes 20
    python 04_topic_model.py --verbose
"""

import argparse
import logging
import sys
import time
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from gensim.corpora import Dictionary
    from gensim.models import LdaMulticore
    from gensim.models import CoherenceModel
except ImportError:
    print(
        "gensim is required but not installed. Run:\n"
        "  pip install gensim",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "bow_corpus.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "lda"

DEFAULT_K_VALUES = [5, 10, 15, 20, 25]
DEFAULT_PASSES = 15
DEFAULT_ITERATIONS = 50
DEFAULT_RANDOM_STATE = 42
DEFAULT_NO_BELOW = 5
DEFAULT_NO_ABOVE = 0.5

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================================
# Topic Modeler
# ============================================================================

class TopicModeler:
    """LDA topic modeling on a lemmatized token corpus."""

    def __init__(
        self,
        no_below: int = DEFAULT_NO_BELOW,
        no_above: float = DEFAULT_NO_ABOVE,
        passes: int = DEFAULT_PASSES,
        iterations: int = DEFAULT_ITERATIONS,
        random_state: int = DEFAULT_RANDOM_STATE,
        workers: Optional[int] = None,
    ):
        self.no_below = no_below
        self.no_above = no_above
        self.passes = passes
        self.iterations = iterations
        self.random_state = random_state
        self.workers = workers or max(1, cpu_count() - 1)

    def prepare_corpus(
        self, token_lists: List[List[str]]
    ) -> Tuple[Dictionary, List[List[Tuple[int, int]]]]:
        dictionary = Dictionary(token_lists)
        vocab_before = len(dictionary)

        dictionary.filter_extremes(
            no_below=self.no_below, no_above=self.no_above
        )
        vocab_after = len(dictionary)
        logger.info(
            f"Dictionary: {vocab_before:,} -> {vocab_after:,} terms "
            f"(filtered {vocab_before - vocab_after:,})"
        )

        if vocab_after == 0:
            logger.error(
                "All tokens filtered out. Try lowering --no-below or raising --no-above."
            )
            sys.exit(1)

        bow_corpus = [dictionary.doc2bow(tokens) for tokens in token_lists]
        return dictionary, bow_corpus

    def train_model(
        self,
        dictionary: Dictionary,
        bow_corpus: List,
        num_topics: int,
    ) -> LdaMulticore:
        logger.info(f"Training LDA with k={num_topics}...")
        start = time.time()
        model = LdaMulticore(
            corpus=bow_corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=self.passes,
            iterations=self.iterations,
            random_state=self.random_state,
            workers=self.workers,
        )
        elapsed = time.time() - start
        logger.info(f"  Trained in {elapsed:.1f}s")
        return model

    def compute_coherence(
        self,
        model: LdaMulticore,
        token_lists: List[List[str]],
        dictionary: Dictionary,
    ) -> float:
        cm = CoherenceModel(
            model=model,
            texts=token_lists,
            dictionary=dictionary,
            coherence="c_v",
        )
        score = cm.get_coherence()
        return score

    def get_document_topics(
        self,
        model: LdaMulticore,
        bow_corpus: List,
        num_topics: int,
    ) -> pd.DataFrame:
        rows = []
        for bow in bow_corpus:
            topic_dist = model.get_document_topics(
                bow, minimum_probability=0.0
            )
            probs = [0.0] * num_topics
            for topic_id, prob in topic_dist:
                probs[topic_id] = prob
            rows.append(probs)

        columns = [f"topic_{i}" for i in range(num_topics)]
        df = pd.DataFrame(rows, columns=columns)
        df["dominant_topic"] = df.values.argmax(axis=1)
        return df

    def find_best_k(
        self,
        k_values: List[int],
        dictionary: Dictionary,
        bow_corpus: List,
        token_lists: List[List[str]],
    ) -> Dict:
        scores = []
        models = {}

        for k in k_values:
            model = self.train_model(dictionary, bow_corpus, k)
            coherence = self.compute_coherence(model, token_lists, dictionary)
            scores.append((k, coherence))
            models[k] = model
            logger.info(f"  k={k}: coherence={coherence:.4f}")

            top_words = model.print_topics(num_words=5)
            for topic_id, words in top_words:
                logger.debug(f"    Topic {topic_id}: {words}")

        best_k, best_score = max(scores, key=lambda x: x[1])
        logger.info(f"\nBest k={best_k} (coherence={best_score:.4f})")

        return {
            "scores": scores,
            "best_k": best_k,
            "best_score": best_score,
            "models": models,
        }

    @staticmethod
    def print_topics(model: LdaMulticore, num_words: int = 10) -> None:
        topics = model.show_topics(
            num_topics=-1, num_words=num_words, formatted=False
        )
        for topic_id, words in topics:
            word_str = ", ".join(f"{w} ({p:.3f})" for w, p in words)
            logger.info(f"  Topic {topic_id:2d}: {word_str}")


# ============================================================================
# Visualization
# ============================================================================

def plot_coherence(
    scores: List[Tuple[int, float]], output_path: Path
) -> None:
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not installed, skipping coherence plot")
        return

    k_vals = [s[0] for s in scores]
    coherences = [s[1] for s in scores]
    best_idx = coherences.index(max(coherences))

    colors = ["#4C72B0"] * len(k_vals)
    colors[best_idx] = "#C44E52"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(k_vals)), coherences, color=colors, tick_label=k_vals)
    ax.set_xlabel("Number of Topics (k)")
    ax.set_ylabel("Coherence Score (c_v)")
    ax.set_title("LDA Topic Model Coherence")

    for i, v in enumerate(coherences):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Coherence plot saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LDA topic modeling on lemmatized testimony corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=DEFAULT_INPUT,
        help=f"Input parquet (default: {DEFAULT_INPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--num-topics", "-k", type=int, nargs="+", default=DEFAULT_K_VALUES,
        help=f"Topic counts to try (default: {DEFAULT_K_VALUES})",
    )
    parser.add_argument(
        "--passes", type=int, default=DEFAULT_PASSES,
        help=f"LDA training passes (default: {DEFAULT_PASSES})",
    )
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS,
        help=f"Iterations per pass (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--no-below", type=int, default=DEFAULT_NO_BELOW,
        help=f"Min docs for a token to be kept (default: {DEFAULT_NO_BELOW})",
    )
    parser.add_argument(
        "--no-above", type=float, default=DEFAULT_NO_ABOVE,
        help=f"Max doc fraction for a token (default: {DEFAULT_NO_ABOVE})",
    )
    parser.add_argument(
        "--random-state", type=int, default=DEFAULT_RANDOM_STATE,
        help=f"Random seed (default: {DEFAULT_RANDOM_STATE})",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel workers (default: cpu_count - 1)",
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
    logger.info("LDA TOPIC MODELING")
    logger.info("=" * 60)
    logger.info(f"Input:         {args.input}")
    logger.info(f"Output dir:    {args.output_dir}")
    logger.info(f"k values:      {args.num_topics}")
    logger.info(f"Passes:        {args.passes}")
    logger.info(f"Iterations:    {args.iterations}")
    logger.info(f"Filter:        no_below={args.no_below}, no_above={args.no_above}")
    logger.info(f"Random state:  {args.random_state}")

    # ---- Load ----
    logger.info("\nLoading data...")
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} documents")

    token_lists = [
        arr.tolist() if isinstance(arr, np.ndarray) else list(arr)
        for arr in df["tokens"]
    ]

    empty_count = sum(1 for t in token_lists if len(t) == 0)
    if empty_count:
        logger.warning(f"{empty_count} document(s) have 0 tokens")

    # ---- Prepare corpus ----
    logger.info("\n" + "=" * 60)
    logger.info("BUILDING DICTIONARY & CORPUS")
    logger.info("=" * 60)

    modeler = TopicModeler(
        no_below=args.no_below,
        no_above=args.no_above,
        passes=args.passes,
        iterations=args.iterations,
        random_state=args.random_state,
        workers=args.workers,
    )

    dictionary, bow_corpus = modeler.prepare_corpus(token_lists)

    # ---- Train ----
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING LDA MODELS")
    logger.info("=" * 60)

    k_values = sorted(args.num_topics)

    if len(k_values) == 1:
        k = k_values[0]
        model = modeler.train_model(dictionary, bow_corpus, k)
        coherence = modeler.compute_coherence(model, token_lists, dictionary)
        scores = [(k, coherence)]
        best_k = k
        logger.info(f"Coherence (c_v): {coherence:.4f}")
    else:
        result = modeler.find_best_k(k_values, dictionary, bow_corpus, token_lists)
        scores = result["scores"]
        best_k = result["best_k"]
        model = result["models"][best_k]

    # ---- Print topics ----
    logger.info("\n" + "=" * 60)
    logger.info(f"TOPICS (k={best_k})")
    logger.info("=" * 60)
    modeler.print_topics(model, num_words=10)

    # ---- Document-topic matrix ----
    logger.info("\n" + "=" * 60)
    logger.info("DOCUMENT-TOPIC DISTRIBUTIONS")
    logger.info("=" * 60)

    doc_topics = modeler.get_document_topics(model, bow_corpus, best_k)
    doc_topics.insert(0, "doc_id", df["doc_id"].values)

    topic_counts = doc_topics["dominant_topic"].value_counts().sort_index()
    logger.info("Documents per dominant topic:")
    for topic_id, count in topic_counts.items():
        logger.info(f"  Topic {topic_id}: {count} docs")

    # ---- Save ----
    logger.info("\n" + "=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)

    model_dir = args.output_dir / "lda_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir / "lda.model"))
    dictionary.save(str(model_dir / "dictionary.dict"))
    logger.info(f"Model saved to {model_dir}/")

    topics_path = args.output_dir / "lda_topics.parquet"
    doc_topics.to_parquet(topics_path, index=False)
    logger.info(f"Document-topic matrix saved to {topics_path}")

    coherence_path = args.output_dir / "lda_coherence.csv"
    pd.DataFrame(scores, columns=["k", "coherence"]).to_csv(
        coherence_path, index=False
    )
    logger.info(f"Coherence scores saved to {coherence_path}")

    plot_path = args.output_dir / "lda_coherence.png"
    if len(scores) > 1:
        plot_coherence(scores, plot_path)

    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
