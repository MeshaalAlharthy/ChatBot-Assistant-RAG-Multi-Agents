import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

from functools import lru_cache

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from rag_assistant.config import CHROMA_DIR, COLLECTION, EMBED_MODEL, LEVELS


@lru_cache(maxsize=1)
def get_store() -> Chroma:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError("No index yet. Run: python -m rag_assistant.ingest")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR),
                                       settings=Settings(anonymized_telemetry=False))
    try:
        client.get_collection(COLLECTION)
    except Exception:
        raise FileNotFoundError(f"No index named '{COLLECTION}' yet. Run: python -m rag_assistant.ingest")
    return Chroma(client=client, collection_name=COLLECTION,
                  embedding_function=OllamaEmbeddings(model=EMBED_MODEL))


def access_filter(clearance: str, allow_pii: bool) -> dict:
    if clearance not in LEVELS:
        raise ValueError(f"unknown clearance: {clearance!r}")
    rules = [{"level_rank": {"$lte": LEVELS[clearance][0]}}]
    if not allow_pii:
        rules.append({"has_pii": False})
    return rules[0] if len(rules) == 1 else {"$and": rules}


def retrieve(question: str, clearance: str = "public", allow_pii: bool = False,
             k: int = 4) -> list[tuple[Document, float]]:
    flt = access_filter(clearance, allow_pii)
    results = get_store().similarity_search_with_score(question, k=k, filter=flt)
    return [(doc, 1 - distance) for doc, distance in results]
