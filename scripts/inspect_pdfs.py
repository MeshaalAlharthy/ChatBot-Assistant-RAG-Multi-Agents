import re
import sys
from pathlib import Path

from rag_assistant.config import DOCS_DIR, LEVELS, level_of
from rag_assistant.pdf_text import extract_pages
from rag_assistant.pii import find_pii, mask

COMMON = ["في", "من", "على", "إلى", "التي", "الذي", "عن", "مع", "هذا", "هذه"]
PAIRS = ["من خلال", "يجب أن", "في حال", "من قبل", "على أن", "بناء على"]
BROKEN_LAM_ALEF = re.compile(r"ا[اأإآ]ل")
ARABIC = re.compile(r"[؀-ۿ]")


def quality(text: str) -> str:
    words = re.findall(r"\w+", text)
    flat = " ".join(text.split())
    if not ARABIC.search(text):
        return "no Arabic text"
    if sum(words.count(w[::-1]) for w in COMMON) > sum(words.count(w) for w in COMMON):
        return "REVERSED letters"
    reversed_pairs = [" ".join(p.split()[::-1]) for p in PAIRS]
    if sum(flat.count(p) for p in reversed_pairs) > sum(flat.count(p) for p in PAIRS):
        return "REVERSED word order"
    broken = len(BROKEN_LAM_ALEF.findall(text))
    if broken:
        return f"BROKEN لا still left ({broken})"
    return "OK"


def inspect(path: Path) -> dict:
    pages, cleaning = extract_pages(path)
    text = "\n".join(pages)
    middle = pages[len(pages) // 2] if pages else ""
    return {
        "pages": len(pages),
        "empty_pages": sum(1 for p in pages if len(p.strip()) < 20),
        "cleaning": cleaning,
        "quality": quality(text),
        "pii": find_pii(text),
        "sample": " ".join(mask(middle).split())[:250],
    }


def main() -> None:
    files = sorted(p for p in DOCS_DIR.rglob("*") if p.suffix.lower() == ".pdf")
    if not files:
        sys.exit(f"No PDFs found in {DOCS_DIR}\n"
                 f"Put them in a level folder, e.g. {DOCS_DIR / 'public'}, and run again.")

    total_pages = 0
    problems, with_pii, unplaced = [], [], []
    per_level = {name: 0 for name in LEVELS}
    for i, path in enumerate(files, 1):
        r = inspect(path)
        total_pages += r["pages"]
        level = level_of(path)
        if level:
            per_level[level] += 1
        else:
            unplaced.append(path.name)
        pii_found = {k: v for k, v in r["pii"].items() if v}
        if r["quality"] != "OK":
            problems.append(path.name)
        if pii_found:
            with_pii.append(path.name)

        print(f"\n[{i}/{len(files)}] {path.relative_to(DOCS_DIR)}")
        if level:
            print(f"   level: {level} ({LEVELS[level][1]})")
        else:
            print("   level: WARNING: NOT IN A LEVEL FOLDER -> will NOT be indexed")
        print(f"   pages: {r['pages']}  (no text: {r['empty_pages']})")
        print(f"   fixed: {r['cleaning']['lam_alef_fixes']} reversed 'لا'")
        print(f"   removed header/footer lines: {r['cleaning']['removed_lines'] or 'none'}")
        print(f"   removed repeated table cells: {r['cleaning']['duplicate_lines']}")
        for sentence in r["cleaning"]["injections"]:
            print(f"   WARNING: removed instruction aimed at the AI: {sentence[:90]}")
        print(f"   text quality: {r['quality']}")
        print(f"   personal data: {pii_found or 'none found'}")
        print(f"   sample: {r['sample']}")

    print("\n" + "=" * 60)
    print(f"Files: {len(files)}   Pages: {total_pages}")
    print("By level: " + ", ".join(f"{k}={v}" for k, v in per_level.items()))
    print(f"Not in a level folder (won't be indexed): {len(unplaced)} {unplaced or ''}")
    print(f"Files with text problems: {len(problems)} {problems or ''}")
    print(f"Files with possible personal data: {len(with_pii)} {with_pii or ''}")


if __name__ == "__main__":
    main()
