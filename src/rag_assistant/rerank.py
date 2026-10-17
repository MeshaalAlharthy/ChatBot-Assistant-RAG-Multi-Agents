import argparse
import logging
import os
import time
from functools import lru_cache

from rag_assistant.config import RERANK_MODEL, RERANK_REPO, RERANK_TOP_N, RETRIEVE_K

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_reranker():
    if not RERANK_MODEL.exists():
        log.warning("reranker not found in %s. Run: python -m rag_assistant.rerank --download", RERANK_MODEL)
        return None
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(str(RERANK_MODEL), max_length=512, local_files_only=True)
    except Exception as e:
        log.warning("reranker could not be loaded (%s); keeping the search order", e)
        return None


def rerank(question: str, results: list, top_n: int = RERANK_TOP_N) -> tuple[list, str]:
    if not results:
        return [], "no candidates"
    model = get_reranker()
    if model is None:
        return results[:top_n], "model not available, search order kept"

    t0 = time.perf_counter()
    scores = model.predict([(question, doc.page_content) for doc, _ in results])
    ranked = sorted(zip(results, scores), key=lambda pair: float(pair[1]), reverse=True)
    kept = []
    for (doc, similarity), score in ranked[:top_n]:
        doc.metadata["rerank_score"] = round(float(score), 3)
        kept.append((doc, similarity))

    search_top = {id(doc) for doc, _ in results[:top_n]}
    promoted = sum(id(doc) not in search_top for doc, _ in kept)
    return kept, (f"{len(results)} candidates -> best {len(kept)}, {promoted} brought up "
                  f"from lower in the list ({time.perf_counter() - t0:.1f}s)")


def download() -> None:
    from huggingface_hub import snapshot_download
    print(f"Downloading {RERANK_REPO} to {RERANK_MODEL} ...")
    snapshot_download(RERANK_REPO, local_dir=str(RERANK_MODEL),
                      ignore_patterns=["*.onnx", "onnx/*", "*.h5", "*.msgpack", "*.ot"])
    print("Done. The app loads it from this folder, without internet.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--download", action="store_true", help="download the model into models/ (once)")
    ap.add_argument("--clearance", choices=["public", "restricted", "secret", "top_secret"], default="public")
    ap.add_argument("--allow-pii", action="store_true")
    args = ap.parse_args()
    if args.download:
        download()
        return
    if not args.question:
        ap.error("a question is required (or use --download)")

    from rag_assistant.rag import keep_relevant
    from rag_assistant.retriever import retrieve

    candidates = keep_relevant(retrieve(args.question, args.clearance, args.allow_pii, k=RETRIEVE_K))
    print(f"\nSearch order ({len(candidates)} candidates above the similarity threshold):")
    for i, (doc, sim) in enumerate(candidates, 1):
        print(f"  {i:>2}. similarity {sim:.2f} | {doc.metadata['source']} p.{doc.metadata['page']}")
    kept, note = rerank(args.question, candidates)
    print(f"\nAfter reranking: {note}")
    for i, (doc, sim) in enumerate(kept, 1):
        print(f"  {i:>2}. rerank {doc.metadata.get('rerank_score', '-')} | similarity {sim:.2f} | "
              f"{doc.metadata['source']} p.{doc.metadata['page']}")
        print("      " + " ".join(doc.page_content.split())[:160])


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    main()
