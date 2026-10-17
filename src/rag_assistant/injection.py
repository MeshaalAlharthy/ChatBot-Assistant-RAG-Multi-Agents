import re

PATTERNS = [
    r"تجاهل\s+(جميع\s+|كل\s+)?(التعليمات|القواعد|الأوامر)",
    r"(موجهة?|رسالة|ملاحظة|تعليمات)\s+(إلى|الى|لل)\s*(المساعد|النموذج|الذكاء\s+الاصطناعي|الروبوت)",
    r"(إلى|الى)\s+المساعد\s+الآلي",
    r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+(instructions|rules)",
    r"(note|message|instructions?)\s+(to|for)\s+(the\s+)?(ai|assistant|model|llm)\b",
]
INJECTION = re.compile("|".join(PATTERNS), re.IGNORECASE)


SENTENCE_END = re.compile(r"[.؟!:]\s*$")
MAX_LINES = 4


def strip_injections(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines()
    kept, removed, i = [], [], 0
    while i < len(lines):
        if INJECTION.search(lines[i]):
            j = i
            while j < len(lines) - 1 and j - i < MAX_LINES - 1 and not SENTENCE_END.search(lines[j]):
                j += 1
            removed.append(" ".join(lines[i:j + 1]))
            i = j + 1
        else:
            kept.append(lines[i])
            i += 1
    return "\n".join(kept), removed
