import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import shutil
import sys
import time

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_assistant.config import (CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION, DOCS_DIR,
                    EMBED_MODEL, LEVELS, level_of)
from rag_assistant.pdf_text import extract_pages
from rag_assistant.pii import find_pii

BATCH = 32

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "، ", " ", ""],
)


def chunk_pdf(path, level: str) -> list[Document]:
    pages, report = extract_pages(path)
    for sentence in report["injections"]:
        print(f"WARNING: {path.name}: removed instruction aimed at the AI: {sentence[:80]}")
    rank, _ = LEVELS[level]
    docs = []
    for page_no, text in enumerate(pages, start=1):
        for i, chunk in enumerate(splitter.split_text(text)):
            if len(chunk.strip()) < 50:
                continue
            pii = {k: v for k, v in find_pii(chunk).items() if v}
            docs.append(Document(
                page_content=chunk,
                id=f"{level}/{path.name}#p{page_no}-c{i}",
                metadata={
                    "source": path.name,
                    "page": page_no,
                    "level": level,
                    "level_rank": rank,
                    "has_pii": bool(pii),
                    "pii_types": ",".join(sorted(pii)),
                },
            ))
    return docs


def main() -> None:
    files = sorted(p for p in DOCS_DIR.rglob("*") if p.suffix.lower() == ".pdf")
    all_docs = []
    for path in files:
        level = level_of(path)
        if level is None:
            print(f"WARNING: skipped (not in a level folder): {path.relative_to(DOCS_DIR)}")
            continue
        docs = chunk_pdf(path, level)
        pii_chunks = sum(d.metadata["has_pii"] for d in docs)
        print(f"{path.relative_to(DOCS_DIR)}: {len(docs)} chunks"
              + (f"  ({pii_chunks} with personal data)" if pii_chunks else ""))
        all_docs.extend(docs)
    if not all_docs:
        sys.exit(f"Nothing to index in {DOCS_DIR}. Run: python scripts/inspect_pdfs.py")

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR),
                                       settings=Settings(anonymized_telemetry=False))
    store = Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        collection_metadata={"hnsw:space": "cosine"},
    )

    t0 = time.perf_counter()
    for start in range(0, len(all_docs), BATCH):
        batch = all_docs[start:start + BATCH]
        store.add_documents(batch, ids=[d.id for d in batch])
        done = start + len(batch)
        print(f"\rEmbedding: {done}/{len(all_docs)} chunks", end="", flush=True)
    secs = time.perf_counter() - t0

    print(f"\n\nIndex ready: {len(all_docs)} chunks from {len({d.metadata['source'] for d in all_docs})} "
          f"files in {secs:.0f}s  ->  {CHROMA_DIR.relative_to(CHROMA_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
