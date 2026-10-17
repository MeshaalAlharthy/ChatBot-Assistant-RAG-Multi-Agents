import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

from rag_assistant.injection import strip_injections

ALEFS = "اأإآ"
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
MARGIN = 0.12
SHORT_LINE = 30
BULLET_ONLY = re.compile(r"[\s•▪●◦·*\-–—]+")


def _span_text(chars: list[dict]) -> tuple[str, int]:
    out, fixes, i = [], 0, 0
    while i < len(chars):
        c = chars[i]
        x0, _, x1, _ = c["bbox"]
        nxt = chars[i + 1]["c"] if i + 1 < len(chars) else ""
        if c["c"] in ALEFS and abs(x1 - x0) < 0.01 and nxt == "ل":
            out += ["ل", c["c"]]
            fixes += 1
            i += 2
        else:
            out.append(c["c"])
            i += 1
    return "".join(out), fixes


TATWEEL_AND_HARAKAT = re.compile(r"[ـً-ْ]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(DIGITS)
    return TATWEEL_AND_HARAKAT.sub("", text)


def _key(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip())


def _page_lines(page: pymupdf.Page) -> tuple[list[tuple[str, bool]], int]:
    height = page.rect.height
    lines, fixes = [], 0
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            parts = []
            for span in line["spans"]:
                text, n = _span_text(span["chars"])
                parts.append(text)
                fixes += n
            _, y0, _, y1 = line["bbox"]
            in_margin = y1 <= MARGIN * height or y0 >= (1 - MARGIN) * height
            lines.append((normalize("".join(parts)), in_margin))
    return lines, fixes


def extract_pages(path: Path) -> tuple[list[str], dict]:
    with pymupdf.open(path) as doc:
        results = [_page_lines(page) for page in doc]

    seen = Counter()
    for lines, _ in results:
        seen.update({_key(t) for t, in_margin in lines if in_margin and t.strip()})
    repeated = {k for k, n in seen.items() if len(results) >= 4 and n >= 0.5 * len(results)}

    pages, duplicates, injections = [], 0, []
    for lines, _ in results:
        kept, seen = [], set()
        for t, in_margin in lines:
            if in_margin and _key(t) in repeated:
                continue
            if BULLET_ONLY.fullmatch(t):
                continue
            short = t.strip()
            if len(short) <= SHORT_LINE and short in seen:
                duplicates += 1
                continue
            seen.add(short)
            kept.append(t)
        page_text, removed = strip_injections("\n".join(kept))
        injections.extend(removed)
        pages.append(page_text)

    report = {
        "lam_alef_fixes": sum(n for _, n in results),
        "removed_lines": sorted(repeated),
        "duplicate_lines": duplicates,
        "injections": injections,
    }
    return pages, report
