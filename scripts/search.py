import argparse

from rag_assistant.config import LEVELS
from rag_assistant.retriever import access_filter, retrieve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=4, help="how many chunks to return")
    ap.add_argument("--clearance", choices=list(LEVELS), default="public")
    ap.add_argument("--allow-pii", action="store_true")
    args = ap.parse_args()

    print(f"\nfilter: {access_filter(args.clearance, args.allow_pii)}\n")
    results = retrieve(args.question, args.clearance, args.allow_pii, args.k)
    for rank, (doc, similarity) in enumerate(results, 1):
        m = doc.metadata
        print(f"#{rank}  similarity {similarity:.2f} | {m['source']} p.{m['page']} | {m['level']}"
              + (f" | PII: {m['pii_types']}" if m["has_pii"] else ""))
        print("    " + " ".join(doc.page_content.split())[:300] + "\n")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        raise SystemExit(str(e))
