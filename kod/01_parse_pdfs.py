"""Extract text from Ravensbrück testimony PDFs into structured data.

Usage:
    python 01_parse_pdfs.py
    python 01_parse_pdfs.py --verbose
    python 01_parse_pdfs.py --doc-dir /path/to/pdfs --out-dir /path/to/output
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import pdfplumber

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOC_DIR = Path.home() / "Work" / "Ravensbruck" / "document"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data"

logger = logging.getLogger(__name__)


# ============================================================================
# PDF text extraction
# ============================================================================

def extract_pdf_text(pdf_path: Path) -> dict:
    """Extract all text from a single PDF, returning per-page and combined text."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages: list[str] = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)

            return {
                "filename": pdf_path.name,
                "doc_id": pdf_path.stem,
                "n_pages": len(pages),
                "full_text": "\n\n".join(pages),
                "page_texts": pages,
                "error": None,
            }
    except Exception as e:
        logger.error("Failed to extract %s: %s", pdf_path.name, e)
        return {
            "filename": pdf_path.name,
            "doc_id": pdf_path.stem,
            "n_pages": 0,
            "full_text": "",
            "page_texts": [],
            "error": str(e),
        }


def extract_all(doc_dir: Path) -> list[dict]:
    """Extract text from every PDF in the document directory."""
    pdf_paths = sorted(doc_dir.glob("*.pdf"))
    logger.info("Found %d PDFs in %s", len(pdf_paths), doc_dir)

    results = []
    for i, path in enumerate(pdf_paths, 1):
        if i % 50 == 0 or i == len(pdf_paths):
            logger.info("Processing %d / %d: %s", i, len(pdf_paths), path.name)
        results.append(extract_pdf_text(path))

    n_errors = sum(1 for r in results if r["error"])
    logger.info("Extraction complete: %d succeeded, %d failed", len(results) - n_errors, n_errors)
    return results


# ============================================================================
# Output
# ============================================================================

def save_results(results: list[dict], out_dir: Path) -> None:
    """Save extracted texts as parquet (corpus table) and individual .txt files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Corpus table (one row per document) ---
    df = pd.DataFrame([
        {
            "doc_id": r["doc_id"],
            "filename": r["filename"],
            "n_pages": r["n_pages"],
            "n_chars": len(r["full_text"]),
            "n_words": len(r["full_text"].split()),
            "full_text": r["full_text"],
            "error": r["error"],
        }
        for r in results
    ])
    parquet_path = out_dir / "corpus.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved corpus table: %s (%d rows)", parquet_path, len(df))

    # --- Individual text files ---
    txt_dir = out_dir / "txt"
    txt_dir.mkdir(exist_ok=True)
    for r in results:
        if r["full_text"]:
            (txt_dir / f"{r['doc_id']}.txt").write_text(r["full_text"], encoding="utf-8")
    logger.info("Saved %d text files to %s", sum(1 for r in results if r["full_text"]), txt_dir)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Corpus summary")
    print(f"{'='*60}")
    print(f"  Documents:    {len(df)}")
    print(f"  Total pages:  {df['n_pages'].sum()}")
    print(f"  Total words:  {df['n_words'].sum():,}")
    print(f"  Total chars:  {df['n_chars'].sum():,}")
    print(f"  Errors:       {df['error'].notna().sum()}")
    print(f"  Output:       {out_dir}")
    print(f"{'='*60}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-dir", type=Path, default=DEFAULT_DOC_DIR,
                        help="Directory containing testimony PDFs")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Output directory for extracted data")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    results = extract_all(args.doc_dir)
    save_results(results, args.out_dir)


if __name__ == "__main__":
    main()
