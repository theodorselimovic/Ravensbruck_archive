"""Extract structured metadata from Ravensbrück testimony text files.

Usage:
    python 02_extract_metadata.py
    python 02_extract_metadata.py --verbose
    python 02_extract_metadata.py --txt-dir /path/to/txt --out-dir /path/to/output
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TXT_DIR = PROJECT_ROOT / "data" / "txt"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data"

logger = logging.getLogger(__name__)

STAMP_NOISE = {
    "POLISH SOURCE INSTITUTE",
    "IN LUND",
    "[stamp]",
    "[/stamp]",
    "BLOM'S PRINTING, LUND",
    "BLOMS BOKTRYCKERI, LUND",
    "Cont'd overleaf",
}

STAMP_NOISE_PATTERNS = re.compile(
    r"^("
    r"POLISH SOURCE INSTITUTE|IN LUND|"
    r"\[/?stamp\]|"
    r"BLOM.?S PRINTING.*|BLOMS BOKTRYCKERI.*|"
    r"Cont.d overleaf|"
    r"\d{4}"  # bare year like "1945", "1946"
    r")$",
    re.IGNORECASE,
)


# ============================================================================
# Section splitting
# ============================================================================

def find_boundary(text: str, marker: str, start: int = 0, case_sensitive: bool = False) -> int:
    """Find position of a boundary marker in text. Returns -1 if not found."""
    if case_sensitive:
        return text.find(marker, start)
    return text.lower().find(marker.lower(), start)


def find_assessment_boundary(text: str, search_start: int) -> tuple[int, int]:
    """Find the closing line and interviewer assessment start.

    Returns (closing_pos, assessment_start) where closing_pos is where testimony
    body ends and assessment_start is where the assessment text begins.
    Returns (-1, -1) if no boundary found.
    """
    def line_start_of(pos: int) -> int:
        ls = text.rfind("\n", 0, pos)
        return ls + 1 if ls != -1 else pos

    def find_all(marker: str, start: int) -> list[int]:
        positions = []
        pos = start
        while True:
            p = find_boundary(text, marker, pos)
            if p == -1:
                break
            positions.append(p)
            pos = p + 1
        return positions

    def skip_closing_lines(pos: int) -> int:
        """Skip past closing line + 1-2 name/role lines after 'Read and signed'."""
        line_end = text.find("\n", pos)
        if line_end == -1:
            return len(text)
        # Skip next line (usually witness name + role)
        next_nl = text.find("\n", line_end + 1)
        if next_nl == -1:
            return line_end + 1
        # Check if the line after that is also a short name/role line
        line_after = text[next_nl + 1:text.find("\n", next_nl + 1) if text.find("\n", next_nl + 1) != -1 else len(text)].strip()
        if line_after and len(line_after) < 40 and not any(
            line_after.lower().startswith(p) for p in
            ("the witness", "the testifier", "the testimony", "opinion", "comment", "remark")
        ):
            next_nl2 = text.find("\n", next_nl + 1)
            return next_nl2 + 1 if next_nl2 != -1 else next_nl + 1
        return next_nl + 1

    best_closing = -1
    best_assessment = -1

    # --- Pattern 1: "Read, signed" (with comma) — last occurrence ---
    for pos in find_all("ead, signed", search_start):
        ls = line_start_of(pos)
        prefix = text[ls:pos].strip()
        if len(prefix) > 20:
            best_closing = ls
            best_assessment = ls
        else:
            best_closing = ls
            best_assessment = skip_closing_lines(ls)

    # --- Pattern 2: "Read and signed" (no comma) — last occurrence ---
    # Only match when near line start or preceded by a known closing label
    if best_closing == -1:
        for pos in find_all("ead and signed", search_start):
            ls = line_start_of(pos)
            prefix = text[ls:pos].strip().lower()
            is_closing = (
                len(prefix) <= 5
                or "testimony received by" in prefix
                or "testimony recorded by" in prefix
                or "recorded by" in prefix
                or "finished by" in prefix
            )
            if is_closing:
                best_closing = ls
                best_assessment = skip_closing_lines(ls)

    # --- Pattern 2b: "After reading" / "After interview" ---
    if best_closing == -1:
        for marker in ("after reading", "after interview"):
            for pos in find_all(marker, search_start):
                ls = line_start_of(pos)
                best_closing = ls
                best_assessment = skip_closing_lines(ls)

    # --- Pattern 3: Direct assessment markers ---
    if best_closing == -1:
        for marker in ("Opinion.", "Opinion:", "Opinion ", "Commentary.", "Commentary:"):
            p = text.find(marker, search_start)
            if p != -1:
                ls = line_start_of(p)
                if best_closing == -1 or ls < best_closing:
                    best_closing = ls
                    best_assessment = ls

    # --- Pattern 4: "Comments of/from/by" and "Institute Assistant's comments" ---
    if best_closing == -1:
        comment_patterns = [
            r"Comments?\s+of\s+(?:Institute\s+Assistant|the\s+receiver)",
            r"Comments?\s+from\s+",
            r"Comments?\s+by\s+",
            r"Institute\s+Assistant.s\s+comments?\s*:",
            r"Remarks?\s*[.:]",
            r"Remarks?\s+of\s+",
        ]
        for pat in comment_patterns:
            m = re.search(pat, text[search_start:], re.IGNORECASE)
            if m:
                abs_pos = search_start + m.start()
                ls = line_start_of(abs_pos)
                if best_closing == -1 or ls < best_closing:
                    best_closing = ls
                    best_assessment = ls

    # --- Pattern 5: standalone "comments:" ---
    if best_closing == -1:
        positions = find_all("comments:", search_start)
        if positions:
            ls = line_start_of(positions[-1])
            best_closing = ls
            best_assessment = ls

    # --- Pattern 6: "Remarks" without punctuation ---
    if best_closing == -1:
        positions = find_all("remarks", search_start)
        if positions:
            ls = line_start_of(positions[-1])
            best_closing = ls
            best_assessment = ls

    # --- Pattern 7: Near-end assessment starters (short tail) ---
    if best_closing == -1:
        tail_start = max(search_start, len(text) - 2000)
        assessment_starters = [
            r"The witness\b",
            r"The testifier\b",
            r"The testimony\b",
            r"This testimony\b",
            r"The credibility\b",
            r"The details of the testimony\b",
            r"The writer of\b",
            r"This description\b",
            r"Testimony given\b",
            r"I have no reservations\b",
            r"Reliable witness\b",
            r"I am including this testimony\b",
            r"Strictly according to\b",
            r"For this reason, there is no commentary\b",
            r"Reservations must be expressed\b",
            r"Despite being\b",
        ]
        for pat in assessment_starters:
            m = re.search(pat, text[tail_start:], re.IGNORECASE)
            if m:
                abs_pos = tail_start + m.start()
                ls = line_start_of(abs_pos)
                if best_closing == -1 or ls < best_closing:
                    best_closing = ls
                    best_assessment = ls

    # --- Pattern 8: Long assessment starters (anchored to line start, larger window) ---
    if best_closing == -1:
        tail_start_long = max(search_start, len(text) - 20000)
        anchored_starters = [
            r"^The witness.s testimony\b",
            r"^The witness provides\b",
            r"^The witness describes\b",
            r"^The memoirs of\b",
        ]
        for pat in anchored_starters:
            m = re.search(pat, text[tail_start_long:], re.MULTILINE | re.IGNORECASE)
            if m:
                abs_pos = tail_start_long + m.start()
                ls = line_start_of(abs_pos)
                if best_closing == -1 or ls < best_closing:
                    best_closing = ls
                    best_assessment = ls

    return best_closing, best_assessment


def split_sections(text: str, doc_id: str) -> dict:
    """Split document text into major sections using boundary markers."""
    result = {
        "header_block": "",
        "here_stands_block": "",
        "internment_block": "",
        "content_summary": "",
        "testimony_body": "",
        "interviewer_assessment": "",
        "warnings": [],
    }

    # --- 1. Find "Here stands" ---
    here_pos = find_boundary(text, "Here stands")
    if here_pos == -1:
        result["warnings"].append("No 'Here stands' found")
        logger.warning("%s: No 'Here stands' found", doc_id)
        here_pos = 0

    # --- 2. Find oath clause ---
    oath_marker = "who – having been cautioned"
    oath_pos = find_boundary(text, oath_marker, here_pos)
    if oath_pos == -1:
        oath_marker = "who – having been cautioned"
        oath_pos = find_boundary(text, oath_marker, here_pos)
    if oath_pos == -1:
        oath_marker = "who - having been cautioned"
        oath_pos = find_boundary(text, oath_marker, here_pos)
    if oath_pos == -1:
        # Handle line-break between "who" and "– having"
        m = re.search(r"who\s*\n\s*[–\-]\s*having been cautioned", text[here_pos:], re.IGNORECASE)
        if m:
            oath_pos = here_pos + m.start()
        else:
            result["warnings"].append("No oath clause found")
            logger.warning("%s: No oath clause found", doc_id)

    # --- 3. Find "hereby declares" (end of oath, start of internment) ---
    search_from = oath_pos if oath_pos != -1 else here_pos
    declares_marker = "hereby declares as follows:"
    declares_pos = find_boundary(text, declares_marker, search_from)
    if declares_pos == -1:
        declares_marker = "has testified as follows:"
        declares_pos = find_boundary(text, declares_marker, search_from)
    if declares_pos == -1:
        # Handle hyphenated line break: "de-\nclares"
        m = re.search(r"hereby\s+de-\s*clares\s+as\s+follows:", text[search_from:], re.IGNORECASE)
        if m:
            declares_pos = search_from + m.start()
            declares_marker = m.group(0)
    if declares_pos != -1:
        idx = text.lower().find(declares_marker.lower(), search_from)
        declares_end = idx + len(declares_marker) if idx != -1 else declares_pos + len(declares_marker)
    else:
        declares_end = search_from
        result["warnings"].append("No 'declares as follows' found")

    # --- 4. Find "Asked whether" ---
    asked_pos = find_boundary(text, "Asked whether", declares_end if declares_end > 0 else 0)
    if asked_pos == -1:
        result["warnings"].append("No 'Asked whether' found")
        logger.warning("%s: No 'Asked whether' found", doc_id)

    # Find end of "Asked whether" block (the "I state as follows:" line)
    asked_end = asked_pos
    if asked_pos != -1:
        state_pos = find_boundary(text, "I state as follows:", asked_pos)
        if state_pos != -1:
            asked_end = state_pos + len("I state as follows:")
        else:
            # Fallback: take 4 lines from "Asked whether"
            newline_count = 0
            pos = asked_pos
            while pos < len(text) and newline_count < 5:
                if text[pos] == "\n":
                    newline_count += 1
                pos += 1
            asked_end = pos

    # --- 5. Find testimony body start after "Asked whether" ---
    # Use whichever comes first: [stamp] block or BLOM'S PRINTING
    stamp_search_start = asked_end if asked_end > 0 else 0
    stamp_block_pos = find_stamp_block(text, stamp_search_start)
    blom_pos = find_blom_boundary(text, stamp_search_start)

    body_boundary_is_blom = False
    if stamp_block_pos != -1 and blom_pos != -1:
        # If [stamp] comes within ~600 chars after BLOM, the BLOM is a page footer
        # and [stamp] is the real body start. Otherwise, prefer whichever comes first.
        if blom_pos < stamp_block_pos and (stamp_block_pos - blom_pos) < 600:
            stamp_pos = stamp_block_pos
        elif blom_pos < stamp_block_pos:
            stamp_pos = blom_pos
            body_boundary_is_blom = True
        else:
            stamp_pos = stamp_block_pos
    elif stamp_block_pos != -1:
        stamp_pos = stamp_block_pos
    elif blom_pos != -1:
        stamp_pos = blom_pos
        body_boundary_is_blom = True
    else:
        # Last fallback: "Testimony of" line marks body start
        testimony_of_pos = find_boundary(text, "Testimony of", stamp_search_start)
        if testimony_of_pos != -1:
            stamp_pos = testimony_of_pos
        else:
            stamp_pos = -1
            result["warnings"].append("No testimony body boundary found after 'Asked whether'")
            logger.warning("%s: No testimony body boundary found after 'Asked whether'", doc_id)

    # --- 6. Find closing line and interviewer assessment ---
    closing_pos = -1
    assessment_start = -1
    search_start = stamp_pos if stamp_pos != -1 else asked_end

    if search_start > 0:
        closing_pos, assessment_start = find_assessment_boundary(text, search_start)

    # --- Assemble sections ---
    result["header_block"] = text[:here_pos].strip() if here_pos > 0 else ""

    if oath_pos != -1:
        result["here_stands_block"] = text[here_pos:oath_pos].strip()
    elif declares_end > here_pos:
        result["here_stands_block"] = text[here_pos:declares_end].strip()
    else:
        result["here_stands_block"] = ""

    if declares_end > 0 and asked_pos != -1:
        result["internment_block"] = text[declares_end:asked_pos].strip()
    elif declares_end > 0:
        result["internment_block"] = text[declares_end:].strip()[:500]

    if asked_end > 0 and stamp_pos != -1:
        raw_summary = text[asked_end:stamp_pos].strip()
        result["content_summary"] = clean_summary(raw_summary)
    elif asked_end > 0:
        result["content_summary"] = ""

    if stamp_pos != -1:
        if body_boundary_is_blom:
            body_start = find_blom_boundary_end(text, stamp_pos)
        else:
            stamp_block_end = find_stamp_block_end(text, stamp_pos)
            body_start = stamp_block_end if stamp_block_end != -1 else stamp_pos

        # Skip any remaining preamble (BLOM lines, extra stamps, page numbers)
        body_start = skip_preamble(text, body_start)

        if closing_pos != -1 and closing_pos > body_start:
            raw_body = text[body_start:closing_pos].strip()
        else:
            raw_body = text[body_start:].strip()
        result["testimony_body"] = clean_testimony_body(clean_body_start(raw_body))

        if assessment_start != -1 and assessment_start < len(text):
            raw_assessment = text[assessment_start:].strip()
            result["interviewer_assessment"] = clean_assessment(raw_assessment)

    return result


def find_stamp_block(text: str, start: int) -> int:
    """Find the first [stamp] or standalone POLISH SOURCE INSTITUTE block."""
    # Find first [stamp] block with POLISH SOURCE INSTITUTE
    stamp_result = -1
    pos = start
    while True:
        stamp_pos = find_boundary(text, "[stamp]", pos)
        if stamp_pos == -1:
            break
        after_stamp = text[stamp_pos:stamp_pos + 200]
        if "POLISH SOURCE INSTITUTE" in after_stamp.upper():
            stamp_result = stamp_pos
            break
        pos = stamp_pos + 1

    # Also find standalone "POLISH SOURCE INSTITUTE" without [stamp]
    psi_result = -1
    pos = start
    while True:
        psi_pos = find_boundary(text, "Polish Source Institute", pos)
        if psi_pos == -1:
            break
        line_start = text.rfind("\n", 0, psi_pos)
        line = text[line_start + 1:psi_pos + 50] if line_start != -1 else text[:psi_pos + 50]
        line_lower = line.strip().lower()
        if line_lower.startswith("polish source institute") or line_lower.startswith("[stamp]"):
            psi_result = psi_pos if line_start == -1 else line_start + 1
            break
        pos = psi_pos + 1

    if stamp_result != -1 and psi_result != -1:
        return min(stamp_result, psi_result)
    return stamp_result if stamp_result != -1 else psi_result


def find_stamp_block_end(text: str, stamp_start: int) -> int:
    """Find the end of a stamp block (after [/stamp] or after IN LUND)."""
    # Look for [/stamp]
    end_stamp = text.find("[/stamp]", stamp_start)
    if end_stamp != -1 and end_stamp - stamp_start < 200:
        next_nl = text.find("\n", end_stamp)
        return next_nl + 1 if next_nl != -1 else end_stamp + len("[/stamp]")

    # Fallback: find "IN LUND" after stamp_start
    in_lund = find_boundary(text, "IN LUND", stamp_start)
    if in_lund != -1 and in_lund - stamp_start < 200:
        next_nl = text.find("\n", in_lund)
        return next_nl + 1 if next_nl != -1 else in_lund + len("IN LUND")

    return stamp_start


def find_blom_boundary(text: str, start: int) -> int:
    """Find BLOM’S PRINTING / BLOMS BOKTRYCKERI as a fallback body boundary."""
    pos = text.find("BLOM", start)
    if pos != -1:
        line_start = text.rfind("\n", 0, pos)
        line_end = text.find("\n", pos)
        line = text[line_start + 1:line_end] if line_end != -1 else text[line_start + 1:]
        if "PRINTING" in line or "BOKTRYCKERI" in line:
            return line_start + 1 if line_start != -1 else pos
    return -1


def find_blom_boundary_end(text: str, blom_start: int) -> int:
    """Find end of BLOM'S block (BLOM line + year line + blank line)."""
    pos = blom_start
    # Skip BLOM line
    nl = text.find("\n", pos)
    if nl == -1:
        return len(text)
    pos = nl + 1
    # Skip year line (e.g., "1945")
    nl2 = text.find("\n", pos)
    if nl2 != -1:
        line = text[pos:nl2].strip()
        if re.match(r"^\d{4}$", line):
            pos = nl2 + 1
    # Skip blank lines
    while pos < len(text) and text[pos] == "\n":
        pos += 1
    return pos


def find_last_stamp_block(text: str) -> int:
    """Find the last [stamp] POLISH SOURCE INSTITUTE block in the text."""
    last_pos = -1
    pos = 0
    while True:
        stamp_pos = text.find("[stamp]", pos)
        if stamp_pos == -1:
            break
        after = text[stamp_pos:stamp_pos + 200]
        if "POLISH SOURCE INSTITUTE" in after.upper():
            last_pos = stamp_pos
        pos = stamp_pos + 1
    return last_pos


def skip_preamble(text: str, start: int) -> int:
    """Skip past BLOM stamps, [stamp] blocks, page numbers, and blank lines at body start."""
    pos = start
    while pos < len(text):
        # Skip blank lines
        while pos < len(text) and text[pos] in ("\n", "\r"):
            pos += 1

        # Check what the next line is
        line_end = text.find("\n", pos)
        if line_end == -1:
            break
        line = text[pos:line_end].strip()

        # Skip stamp noise lines
        if STAMP_NOISE_PATTERNS.match(line):
            pos = line_end + 1
            continue

        # Skip [stamp] ... [/stamp] blocks
        if line == "[stamp]":
            end_stamp = text.find("[/stamp]", pos)
            if end_stamp != -1:
                pos = end_stamp + len("[/stamp]")
                nl = text.find("\n", pos)
                pos = nl + 1 if nl != -1 else pos
                continue

        # Skip bare page numbers (e.g., "1", "1.")
        if re.match(r"^\d{1,2}\.?$", line):
            pos = line_end + 1
            continue

        # Not preamble — this is the real body start
        break

    return pos


STAMP_BLOCK_RE = re.compile(
    r"^("
    r"POLISH SOURCE INSTITUTE.*|IN LUND|"
    r"BLOM.?S PRINTING.*|BLOMS BOKTRYCKERI.*|"
    r"\[/?[Ss]tamp\s*\]|"
    r"Cont.d overleaf|"
    r"Testimony of (?:Ms|Mr|Mrs)\b.*"
    r")$",
    re.IGNORECASE,
)

VERTICAL_NOISE_RE = re.compile(
    r"^[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ\s''.,\-]{1,3}$"
)

SAFE_SHORT_WORDS = {"I", "SS", "SK", "KZ"}


def clean_body_start(raw: str) -> str:
    """Strip stamp/noise artifacts from the beginning of a testimony body."""
    text = re.sub(r"\[/?[Ss]tamp\s*\]\s*", "", raw)
    lines = text.split("\n")
    cleaned = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i < 15:
            if not stripped:
                if cleaned:
                    cleaned.append(line)
                continue
            if STAMP_NOISE_PATTERNS.match(stripped):
                continue
            if not cleaned and re.match(r"^\d{1,2}\.?$", stripped):
                continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def clean_testimony_body(raw: str) -> str:
    """Remove page-break artifacts from the full testimony body.

    Strips: page numbers, stamp blocks (POLISH SOURCE INSTITUTE, BLOM'S),
    witness-name header repeats, and vertical OCR noise (single stray characters).
    """
    # Strip inline stamp text
    raw = re.sub(r"\s*POLISH SOURCE INSTITUTE\b.*", "", raw)
    raw = re.sub(r"\s*BLOM.?S (?:PRINTING|BOKTRYCKERI)\b.*", "", raw)

    lines = raw.split("\n")
    cleaned = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            cleaned.append(line)
            continue

        # Page numbers: bare "3" or "3." or "12."
        if re.match(r"^\d{1,2}\.?$", stripped):
            continue

        # Stamp blocks and witness-name page headers
        if STAMP_BLOCK_RE.match(stripped):
            continue

        # Vertical OCR noise: 1-3 chars that aren't real standalone words
        if VERTICAL_NOISE_RE.match(stripped) and stripped not in SAFE_SHORT_WORDS:
            # Check nearby lines (within 2) for real content — if any exist, this is noise
            nearby = [lines[j].strip() for j in range(max(0, i - 2), min(len(lines), i + 3)) if j != i]
            if any(len(n) > 10 for n in nearby):
                continue

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def clean_summary(raw: str) -> str:
    """Remove page-break artifacts (witness name, BLOM stamp, vertical stamp chars) from summary."""
    lines = raw.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if STAMP_NOISE_PATTERNS.match(stripped):
            continue
        if len(stripped) <= 2 and not stripped[0].isdigit():
            continue
        # Remove lines that look like partial stamp fragments (mixed single chars and spaces)
        if re.match(r"^[A-Z\s/\[\]]{1,5}$", stripped) and len(stripped.replace(" ", "")) <= 3:
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


CLOSING_LINE_RE = re.compile(
    r"^("
    r"Read[,\s].*signed|"
    r"After reading|"
    r"\(-\).*|"
    r"Completed\s*\(-\)|"
    r"[–\-]{2,}"
    r")$",
    re.IGNORECASE,
)

NAME_ONLY_RE = re.compile(
    r"^[A-ZÀ-Ž][a-zà-ž]+(?:\s+[A-ZÀ-Ž][a-zà-ž]+){0,3}$"
)


def clean_assessment(raw: str) -> str:
    """Clean the interviewer assessment section.

    Removes: trailing stamp blocks, repeated cover pages, inline stamps,
    page numbers, leading signature/name lines, and closing-line artifacts.
    """
    # Truncate at repeated cover page
    for marker in ("Testimony received by", "Testimony recorded by",
                    "Record of Witness Testimony", "Here stands Ms",
                    "Here stands Mr"):
        pos = raw.find(marker)
        if pos != -1 and pos > 50:
            raw = raw[:pos]
            break

    # Strip inline stamps
    raw = re.sub(r"\s*POLISH SOURCE INSTITUTE\b.*", "", raw)
    raw = re.sub(r"\s*BLOM.?S (?:PRINTING|BOKTRYCKERI)\b.*", "", raw)

    lines = raw.split("\n")

    # Remove trailing noise
    while lines:
        stripped = lines[-1].strip()
        if not stripped or STAMP_NOISE_PATTERNS.match(stripped):
            lines.pop()
        else:
            break

    # Remove stamp blocks, page numbers, and closing-line artifacts throughout
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        if STAMP_BLOCK_RE.match(stripped):
            continue
        if re.match(r"^\d{1,2}\.?$", stripped):
            continue
        if CLOSING_LINE_RE.match(stripped):
            continue
        cleaned.append(line)

    # Strip leading name-only lines and blank lines
    while cleaned:
        stripped = cleaned[0].strip()
        # Remove [stamp] markers for matching
        stripped_clean = re.sub(r"\s*\[/?stamp\]\s*", "", stripped).strip()
        if not stripped_clean:
            cleaned.pop(0)
            continue
        # Short name-only lines at the very start (e.g. "B. Kurowski", "Nowaczyk")
        if len(stripped_clean) < 40 and NAME_ONLY_RE.match(stripped_clean):
            cleaned.pop(0)
            continue
        # "B. Kurowski" style with initial
        if len(stripped_clean) < 40 and re.match(r"^[A-ZÀ-Ž]\.\s+[A-ZÀ-Ž][a-zà-ž]+", stripped_clean):
            cleaned.pop(0)
            continue
        # Role lines like "Institute Assistant", "Witness"
        if stripped_clean.lower() in ("institute assistant", "witness"):
            cleaned.pop(0)
            continue
        # Place/date lines like "Lund, 3 December 1945"
        if re.match(r"^[A-ZÀ-Ž][a-zà-ž]+,?\s+\d{1,2}\s+\w+\s+\d{4}$", stripped_clean):
            cleaned.pop(0)
            continue
        break

    return "\n".join(cleaned).strip()


# ============================================================================
# Field extraction from "Here stands" block
# ============================================================================

FIELD_LABELS = re.compile(
    r"(?:occupation|profession)\b|"
    r"religion\b|"
    r"nationality\b|"
    r"parents?.?\s*(?:forenames|names|husband)\b|"
    r"identification\s+document\s*:|"
    r"husband\b|"
    r"proof\s+of\s+identity\b|"
    r"last\s+place\s+of\s+residence\b|"
    r"current\s+place\s+of\s+residence\b|"
    r"\bwho\s+[–\-]",
    re.IGNORECASE,
)


def _next_field_boundary(text: str, start: int) -> int:
    """Find position of the next field label after *start*, or end of text."""
    m = FIELD_LABELS.search(text, start)
    return m.start() if m else len(text)


def _strip_annotations(text: str) -> str:
    """Remove [note ...], [/note], [sic], [stamp] etc. so field labels aren't obscured."""
    return re.sub(r"\[/?(?:note|stamp)[^\]]*\]", "", text)


def extract_here_stands(block: str) -> dict:
    """Parse key-value pairs from the 'Here stands' block."""
    fields: dict[str, Optional[str]] = {
        "title": None,
        "name": None,
        "birth_date": None,
        "birthplace": None,
        "occupation": None,
        "religion": None,
        "nationality": None,
        "parents_forenames": None,
        "husband": None,
        "proof_of_identity": None,
        "last_residence_poland": None,
        "current_residence": None,
    }

    text = " ".join(_strip_annotations(block).split())

    # Title and name: "Here stands Ms/Mr/Mrs NAME born on"
    m = re.search(r"Here stands\s+(Ms|Mr|Mrs)\s+(.+?)\s+born on", text)
    if m:
        fields["title"] = m.group(1)
        fields["name"] = m.group(2).strip()
    else:
        m = re.search(r"Here stands\s+(Ms|Mr|Mrs)\s+(.+?)(?:\s+born|\s*$)", text)
        if m:
            fields["title"] = m.group(1)
            fields["name"] = m.group(2).strip()
        else:
            m = re.search(r"Here stands\s+(.+?)\s+born on", text)
            if m:
                fields["name"] = m.group(1).strip()

    # Birth date — stop at "in" (case-insensitive) or any field label
    m = re.search(r"born on\s+(.+?)(?:\s+[Ii]n\s|\s*$)", text)
    if m:
        raw = m.group(1).strip()
        boundary = _next_field_boundary(text, m.start(1))
        if boundary < m.end(1):
            raw = text[m.start(1):boundary].strip()
        fields["birth_date"] = raw.rstrip(" ,.")
    else:
        m = re.search(r"born (?:on\s+)?(.+?)(?:\s+[Ii]n\s)", text)
        if m:
            fields["birth_date"] = m.group(1).strip().rstrip(" ,.")

    # Birthplace: "in PLACE" terminated by the next field label
    m = re.search(r"born (?:on\s+)?.+?\s+[Ii]n\s+(.+)", text)
    if m:
        val_start = m.start(1)
        boundary = _next_field_boundary(text, val_start)
        fields["birthplace"] = text[val_start:boundary].strip().rstrip(" ,.")

    # Occupation
    m = re.search(r"(?:occupation|profession(?:\s+of)?)\s+(?:a\s+)?(.+)", text, re.IGNORECASE)
    if m:
        val_start = m.start(1)
        boundary = _next_field_boundary(text, val_start)
        fields["occupation"] = text[val_start:boundary].strip().rstrip(" ,.")

    # Religion — find the *last* occurrence (some docs have a spurious first one)
    last_religion = None
    for rm in re.finditer(r"religion\s+", text, re.IGNORECASE):
        val_start = rm.end()
        boundary = _next_field_boundary(text, val_start)
        val = text[val_start:boundary].strip().rstrip(" ,.")
        if val and not re.match(r"^[,–\-\s]+$", val) and not val.startswith(","):
            last_religion = val
    if last_religion:
        fields["religion"] = last_religion

    # Nationality
    m = re.search(r"nationality\s+(.+)", text, re.IGNORECASE)
    if m:
        val_start = m.start(1)
        boundary = _next_field_boundary(text, val_start)
        fields["nationality"] = text[val_start:boundary].strip().rstrip(" ,.")

    # Parents' forenames — also handles "parents' husband's forenames" variant
    m = re.search(r"parents?.?\s*(?:(?:husband.s\s+)?forenames|names)\s+(.+)", text, re.IGNORECASE)
    if m:
        val_start = m.start(1)
        boundary = _next_field_boundary(text, val_start)
        fields["parents_forenames"] = text[val_start:boundary].strip().rstrip(" ,.")

    # Identification document — merge into proof_of_identity
    if not fields["proof_of_identity"]:
        m = re.search(r"identification\s+document\s*:\s*(.+)", text, re.IGNORECASE)
        if m:
            val_start = m.start(1)
            boundary = _next_field_boundary(text, val_start)
            fields["proof_of_identity"] = text[val_start:boundary].strip().rstrip(" ,.")

    # Husband
    m = re.search(r"husband\s*[–\-]\s*(.+)", text, re.IGNORECASE)
    if m:
        val_start = m.start(1)
        boundary = _next_field_boundary(text, val_start)
        fields["husband"] = text[val_start:boundary].strip().rstrip(" ,.")

    # Proof of identity
    m = re.search(r"proof of identity provided\s+(.+)", text, re.IGNORECASE)
    if m:
        val_start = m.start(1)
        boundary = _next_field_boundary(text, val_start)
        fields["proof_of_identity"] = text[val_start:boundary].strip().rstrip(" ,.")

    # Last place of residence in Poland
    m = re.search(r"last place of residence(?:\s+in Poland)?\s*,?\s+(.+)", text, re.IGNORECASE)
    if m:
        val_start = m.start(1)
        boundary = _next_field_boundary(text, val_start)
        fields["last_residence_poland"] = text[val_start:boundary].strip().rstrip(" ,.")

    # Current place of residence
    m = re.search(r"current place of residence\s+(.+)", text, re.IGNORECASE)
    if m:
        val_start = m.start(1)
        boundary = _next_field_boundary(text, val_start)
        fields["current_residence"] = text[val_start:boundary].strip().rstrip(" ,.")

    return fields


# ============================================================================
# Header extraction
# ============================================================================

def split_recorder(raw: str) -> tuple[Optional[str], Optional[str]]:
    """Split a recorder string like 'Institute Assistant Helena Miklaszewska' into (role, name)."""
    # Remove trailing noise like "Transcribed"
    cleaned = re.sub(r"\s+Transcribed$", "", raw, flags=re.IGNORECASE).strip()

    # "Institute Assistant NAME" — most common pattern
    m = re.match(r"(Institute Assistant)\s+(.+)", cleaned, re.IGNORECASE)
    if m:
        name = m.group(2).strip().rstrip(",")
        # Strip credential suffixes but keep them out of the name
        name = re.sub(r",?\s*(?:LL\s*M|Master of Law|\[with a Master.s degree in Law\])$", "", name).strip()
        if name and name != "[not completed]":
            return m.group(1), name
        return m.group(1), None

    # "Ms NAME testimony accepted by Institute"
    m = re.match(r"Ms\s+(.+?)\s+testimony accepted by", cleaned, re.IGNORECASE)
    if m:
        return None, m.group(1).strip()

    return None, cleaned if cleaned else None


def extract_header(block: str) -> dict:
    """Parse header fields from the text before 'Here stands'."""
    fields: dict[str, Optional[str]] = {
        "testimony_place": None,
        "testimony_date": None,
        "recorder": None,
        "recorder_role": None,
        "recorder_name": None,
        "testimony_number": None,
        "is_transcribed": False,
        "is_confidential": False,
    }

    lines = [l.strip() for l in block.split("\n") if l.strip()]

    fields["is_transcribed"] = any("transcribed" in l.lower() for l in lines)
    fields["is_confidential"] = any("confidential" in l.lower() for l in lines)

    # Place and date: look for a line with a date pattern (day month year)
    for line in lines:
        m = re.match(r"^(.+?)(\d{1,2}\s+\w+\s+\d{4})$", line)
        if m:
            place = m.group(1).strip().rstrip(",").strip()
            fields["testimony_place"] = place if place else None
            fields["testimony_date"] = m.group(2).strip()
            break
        # Date without place (e.g., "7 February 1947")
        m = re.match(r"^(\d{1,2}\s+\w+\s+\d{4})$", line)
        if m:
            fields["testimony_date"] = m.group(1).strip()
            break

    # Recorder — split into role and name
    for line in lines:
        if "testimony received by" in line.lower() or "testimony recorded by" in line.lower():
            m = re.search(r"(?:received|recorded)\s+by\s+(.+)", line, re.IGNORECASE)
            if m:
                raw = m.group(1).strip()
                fields["recorder"] = raw
                role, name = split_recorder(raw)
                fields["recorder_role"] = role
                fields["recorder_name"] = name
            break

    # Testimony number
    text = " ".join(lines)
    m = re.search(r"Record of [Ww]itness [Tt]estimony\s+(\d+)", text)
    if m:
        fields["testimony_number"] = int(m.group(1))

    return fields


# ============================================================================
# Internment field extraction
# ============================================================================

def extract_internment(block: str) -> dict:
    """Extract arrest info and camp movements from the internment block."""
    fields: dict[str, Optional[str]] = {
        "arrest_place": None,
        "arrest_date": None,
        "prisoner_number": None,
        "triangle_color": None,
        "triangle_letter": None,
    }

    text = " ".join(block.split())

    m = re.search(r"I was arrested in\s+(.+?)\s+on\s+(\d.+?)(?:\s*I was|\s*$)", text)
    if m:
        fields["arrest_place"] = m.group(1).strip()
        fields["arrest_date"] = m.group(2).strip()
    else:
        m = re.search(r"I was arrested in\s+(.+?)\s+on\s+(.+?)(?:\s|$)", text)
        if m:
            fields["arrest_place"] = m.group(1).strip()
            fields["arrest_date"] = m.group(2).strip()

    # Prisoner number from first camp entry
    m = re.search(r"bearing the number\s+(\d[\d\s]*\d|\d+)", text)
    if m:
        fields["prisoner_number"] = m.group(1).strip()

    # Triangle color
    m = re.search(r"wearing a\s+(.+?)\s*-?\s*colou?red\s+triangle", text, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        if val and val not in ("[not completed]", "–", "-", "____"):
            fields["triangle_color"] = val

    # Triangle letter
    m = re.search(r"with the letter\s+['\"]?(.+?)['\"]?(?:\s*[.;]\s|\s+I\s|\s+Afterwards|\s+\d+\)|\s*$)", text, re.IGNORECASE)
    if m:
        val = m.group(1).strip().strip("'\".,;- ")
        if val and val.lower() not in ("not completed", "no", "none", "afterwards"):
            fields["triangle_letter"] = val

    return fields


# ============================================================================
# Main processing
# ============================================================================

def process_document(txt_path: Path) -> dict:
    """Process a single testimony text file and extract all metadata."""
    doc_id = txt_path.stem
    text = txt_path.read_text(encoding="utf-8")

    sections = split_sections(text, doc_id)

    header_fields = extract_header(sections["header_block"])
    bio_fields = extract_here_stands(sections["here_stands_block"])
    internment_fields = extract_internment(sections["internment_block"])

    result = {"doc_id": doc_id, "filename": txt_path.name}
    result.update(header_fields)
    result.update(bio_fields)
    result.update(internment_fields)

    result["internment_block"] = sections["internment_block"]
    result["content_summary"] = sections["content_summary"]
    result["testimony_body"] = sections["testimony_body"]
    result["interviewer_assessment"] = sections["interviewer_assessment"]
    result["warnings"] = "; ".join(sections["warnings"]) if sections["warnings"] else None

    return result


def extract_all(txt_dir: Path) -> list[dict]:
    """Process all text files in the directory."""
    txt_paths = sorted(txt_dir.glob("*.txt"))
    logger.info("Found %d text files in %s", len(txt_paths), txt_dir)

    results = []
    for i, path in enumerate(txt_paths, 1):
        if i % 50 == 0 or i == len(txt_paths):
            logger.info("Processing %d / %d: %s", i, len(txt_paths), path.name)
        results.append(process_document(path))

    n_warnings = sum(1 for r in results if r["warnings"])
    logger.info("Done: %d processed, %d with warnings", len(results), n_warnings)
    return results


# ============================================================================
# Gender Classification
# ============================================================================

KNOWN_FEMALE_NAMES = {
    "Miriam", "Edith", "Edit", "Esther", "Ester", "Ruth", "Margit",
    "Judith", "Ingrid", "Astrid", "Gudrun", "Ruchel", "Rachel",
    "Elisabeth", "Elizabeth", "Carmen", "Ellen", "Helen", "Mignon",
}

KNOWN_MALE_NAMES = {
    "Adam", "Alfred", "Andrzej", "Antoni", "Arkadjusz", "Bartłomiej",
    "Bohdan", "Bolesław", "Bonifacy", "Bronisław", "Czesław", "Drew",
    "Edmund", "Emanuel", "Ernest", "Eugeniusz", "Feliks", "Ferenc",
    "Franciszek", "Genek", "Henryk", "Hersz", "Hieronim", "Izaak",
    "Izrael", "Jakub", "Jan", "Jerzy", "Józef", "Kanty", "Karol",
    "Kazimierz", "Leon", "Lucjan", "Ludwik", "Majer", "Marian",
    "Marjan", "Michał", "Mieczysław", "Natan", "Paweł", "Peter",
    "Piotr", "Roman", "Ryszard", "Stanisław", "Stefan", "Tadeusz",
    "Wacław", "Wincenty", "Władysław", "Włodzimierz", "Wojciech",
    "Zbigniew", "Zdzisław", "Zygmunt",
}

_RE_TITLE = re.compile(r"^(Miss|Mrs\.?|Mr\.?|Ms\.?|M\s+Revd|M\s)\s*", re.IGNORECASE)
_RE_ANNOTATION = re.compile(r"\[.*?\]")


def classify_gender(name: Optional[str]) -> str:
    """Classify gender from a Polish name string.

    Uses title prefixes, known-name lookups, and the Polish convention
    that female first names end in -a. Handles surname-first ordering
    and [sic] annotations.
    """
    if not name or name == "Name Unknown":
        return "unknown"

    raw = str(name).strip()

    # Title-based shortcuts
    if re.match(r"^(Mrs|Miss|Ms)\b", raw, re.IGNORECASE):
        return "female"
    if re.match(r"^Mr[\s,.]", raw):
        return "male"

    # Clean annotations and titles
    clean = _RE_TITLE.sub("", raw)
    clean = _RE_ANNOTATION.sub("", clean)
    clean = re.sub(r"[\s,;!]+$", "", clean).strip()
    clean = re.sub(r"\s+", " ", clean)

    parts = clean.split()
    if not parts:
        return "unknown"

    first = parts[0]

    # If first token is ALL CAPS, likely surname-first
    if first.isupper() and len(parts) > 1:
        first = parts[1]

    # Check all name parts against known-name lists
    for p in parts:
        if p in KNOWN_MALE_NAMES:
            return "male"
        if p in KNOWN_FEMALE_NAMES:
            return "female"

    # If first part doesn't end in -a but a later part does (surname-first)
    if not first.endswith("a") and len(parts) > 1:
        for p in parts[1:]:
            if p.endswith("a") and len(p) > 2 and p[0].isupper():
                # Exclude feminine surname suffixes (-ska, -cka, -wska)
                if not re.search(r"(?:ska|cka|wska|ńska)$", p, re.IGNORECASE):
                    return "female"

    # Polish heuristic: female first names end in -a
    if first.endswith("a") or first.endswith("A"):
        return "female"

    return "male"


def add_gender_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'gender' column derived from the name field."""
    df = df.copy()
    df["gender"] = df["name"].apply(classify_gender)
    logger.info(
        "Gender: %d female, %d male, %d unknown",
        (df["gender"] == "female").sum(),
        (df["gender"] == "male").sum(),
        (df["gender"] == "unknown").sum(),
    )
    return df


# ============================================================================
# Output
# ============================================================================

TEXT_COLS = ["content_summary", "testimony_body", "interviewer_assessment"]


def save_results(results: list[dict], out_dir: Path) -> None:
    """Save extracted metadata as a single parquet file."""
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df = add_gender_column(df)

    parquet_path = out_dir / "metadata.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved %s (%d rows)", parquet_path, len(df))

    csv_path = out_dir / "metadata.csv"
    df.drop(columns=TEXT_COLS).to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    # Summary
    print(f"\n{'='*60}")
    print("Metadata extraction summary")
    print(f"{'='*60}")
    print(f"  Documents processed:  {len(df)}")
    print(f"  With warnings:        {df['warnings'].notna().sum()}")
    print()

    meta_cols = [c for c in df.columns if c not in TEXT_COLS + ["doc_id", "filename", "warnings"]]
    for col in meta_cols:
        if col in ("is_transcribed", "is_confidential"):
            n = df[col].sum()
            print(f"  {col:30s} {n:4d} true")
        else:
            n = df[col].notna().sum() if df[col].dtype == "object" else (df[col] > 0).sum()
            print(f"  {col:30s} {n:4d} / {len(df)}")

    print()
    for col in TEXT_COLS:
        n = (df[col].str.len() > 0).sum()
        avg_len = df.loc[df[col].str.len() > 0, col].str.len().mean() if n > 0 else 0
        print(f"  {col:30s} {n:4d} non-empty (avg {avg_len:.0f} chars)")

    print(f"{'='*60}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txt-dir", type=Path, default=DEFAULT_TXT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    results = extract_all(args.txt_dir)
    save_results(results, args.out_dir)


if __name__ == "__main__":
    main()
