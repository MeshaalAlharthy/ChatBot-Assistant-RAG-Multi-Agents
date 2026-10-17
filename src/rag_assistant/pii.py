import re

PII_PATTERNS = {
    "national_id": re.compile(r"(?<!\d)[12]\d{9}(?!\d)"),
    "phone": re.compile(r"(?<!\d)(?:\+?966[\s-]?|0)5\d(?:[\s-]?\d){7}(?!\d)"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "iban": re.compile(r"\bSA\d{2}(?:\s?[0-9A-Z]{4}){5}\b"),
}
TEN_DIGITS = re.compile(r"(?<!\d)\d{10}(?!\d)")


def find_pii(text: str) -> dict[str, int]:
    counts = {n: len(p.findall(text)) for n, p in PII_PATTERNS.items()}
    for run in TEN_DIGITS.findall(text):
        if PII_PATTERNS["phone"].fullmatch(run) or PII_PATTERNS["national_id"].fullmatch(run):
            continue
        if PII_PATTERNS["phone"].fullmatch(run[::-1]):
            counts["phone"] += 1
        elif PII_PATTERNS["national_id"].fullmatch(run[::-1]):
            counts["national_id"] += 1
    return counts


def mask(text: str, label: str | None = None) -> str:
    for name, pattern in PII_PATTERNS.items():
        text = pattern.sub(label or f"[{name}]", text)
    return TEN_DIGITS.sub(label or "[number]", text)
