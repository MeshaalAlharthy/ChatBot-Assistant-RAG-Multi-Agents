from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "data" / "docs"
CHROMA_DIR = ROOT / "data" / "chroma"

ASSISTANT_NAME = "المساعد الرقمي لمكتب إدارة البيانات"

LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "bge-m3"

TEMPERATURE = 0.2
NUM_CTX = 8192
KEEP_ALIVE = "30m"

MIN_SIMILARITY = 0.45

RETRIEVE_K = 12
RERANK_TOP_N = 4
RERANK_REPO = "BAAI/bge-reranker-v2-m3"
RERANK_MODEL = ROOT / "models" / "bge-reranker-v2-m3"

HISTORY_TURNS = 3
HISTORY_CHARS = 600

COLLECTION = "internal_docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

LEVELS = {
    "public": (0, "عام"),
    "restricted": (1, "مقيد"),
    "secret": (2, "سري"),
    "top_secret": (3, "سري للغاية"),
}


USERS = {
    "employee": {"name": "موظف", "clearance": "public", "can_see_pii": False},
    "hr_officer": {"name": "مسؤول موارد بشرية", "clearance": "restricted", "can_see_pii": True},
    "security_officer": {"name": "مسؤول أمن معلومات", "clearance": "secret", "can_see_pii": True},
}


def level_of(pdf: Path) -> str | None:
    parts = pdf.relative_to(DOCS_DIR).parts
    return parts[0] if len(parts) > 1 and parts[0] in LEVELS else None
