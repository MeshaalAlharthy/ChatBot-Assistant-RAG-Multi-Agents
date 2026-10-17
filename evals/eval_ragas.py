import os

os.environ["RAGAS_DO_NOT_TRACK"] = "true"

import argparse
import asyncio
import json
import re
import statistics
import time
from datetime import date
from pathlib import Path

JUDGE_MODEL = "qwen2.5:14b"
OLLAMA_OPENAI_URL = "http://localhost:11434/v1"
USER = "hr_officer"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVAL_SET = HERE / "eval_set.json"
OUT_DIR = HERE / "results"
ANSWERS = OUT_DIR / "answers.json"
SCORES = OUT_DIR / "scores.json"
SUMMARY = OUT_DIR / "summary.md"

METRICS = ["faithfulness", "factual_correctness", "context_recall"]
CITATION = re.compile(r"\s*\[\d+\]")


def generate_answers(items: list[dict]) -> list[dict]:
    from rag_assistant.graph import ACCESS_DENIED, BLOCKED, REFUSED, ask
    from rag_assistant.rag import NO_ANSWER

    fixed = {NO_ANSWER, ACCESS_DENIED, BLOCKED, REFUSED}
    rows = []
    for item in items:
        t0 = time.perf_counter()
        state = ask(item["question"], USER)
        row = {
            "id": item["id"],
            "question": item["question"],
            "reference": item["reference"],
            "answer": state["final_answer"],
            "contexts": [doc.page_content for doc, _ in state.get("results", [])],
            "answered": state["final_answer"] not in fixed,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        rows.append(row)
        status = "answered" if row["answered"] else "NOT answered"
        print(f"[{item['id']:>2}] {status:12} {row['seconds']:5.1f}s  {item['question']}")
    return rows


def make_metrics() -> dict:
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (ContextPrecisionWithReference, ContextRecall,
                                           FactualCorrectness, Faithfulness)

    client = AsyncOpenAI(base_url=OLLAMA_OPENAI_URL, api_key="ollama")
    judge = llm_factory(JUDGE_MODEL, client=client, temperature=0, max_tokens=2048)
    return {
        "faithfulness": Faithfulness(llm=judge),
        "factual_correctness": FactualCorrectness(llm=judge),
        "context_precision": ContextPrecisionWithReference(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }


async def score_row(metrics: dict, row: dict) -> dict:
    q, ref, ctx = row["question"], row["reference"], row["contexts"]
    answer = CITATION.sub("", row["answer"]).strip()
    calls = {
        "faithfulness": lambda: metrics["faithfulness"].ascore(
            user_input=q, response=answer, retrieved_contexts=ctx),
        "factual_correctness": lambda: metrics["factual_correctness"].ascore(
            response=answer, reference=ref),
        "context_precision": lambda: metrics["context_precision"].ascore(
            user_input=q, reference=ref, retrieved_contexts=ctx),
        "context_recall": lambda: metrics["context_recall"].ascore(
            user_input=q, retrieved_contexts=ctx, reference=ref),
    }
    scores = {}
    for name in METRICS:
        call = calls[name]
        if not row["answered"] and name in ("faithfulness", "factual_correctness"):
            scores[name] = None if name == "faithfulness" else 0.0
        elif not ctx and name.startswith("context"):
            scores[name] = 0.0
        else:
            try:
                scores[name] = round(float((await call()).value), 3)
            except Exception as e:
                scores[name] = None
                print(f"      {name}: judge error ({type(e).__name__}: {str(e)[:100]})")
    return scores


async def score_all(rows: list[dict]) -> list[dict]:
    metrics = make_metrics()
    for row in rows:
        t0 = time.perf_counter()
        row["scores"] = await score_row(metrics, row)
        cells = "  ".join(f"{k[:12]}={'-' if v is None else f'{v:.2f}'}"
                          for k, v in row["scores"].items())
        print(f"[{row['id']:>2}] {cells}  ({time.perf_counter() - t0:.0f}s)")
    return rows


def summarize(rows: list[dict]) -> dict:
    summary = {"date": date.today().isoformat(), "judge": JUDGE_MODEL,
               "questions": len(rows),
               "answer_rate": round(sum(r["answered"] for r in rows) / len(rows), 3),
               "avg_seconds_per_question": round(statistics.mean(r["seconds"] for r in rows), 1)}
    for m in METRICS:
        values = [r["scores"][m] for r in rows if r["scores"].get(m) is not None]
        summary[m] = round(statistics.mean(values), 3) if values else None
        summary[f"{m}_n"] = len(values)
    return summary


def write_report(rows: list[dict], summary: dict) -> None:
    SCORES.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    lines = [f"# RAGAS results ({summary['date']})", "",
             f"{summary['questions']} questions, judge: `{summary['judge']}`, "
             f"answered {summary['answer_rate']:.0%}, "
             f"{summary['avg_seconds_per_question']}s per question on average.", "",
             "| Metric | Score | Questions scored |", "|---|---|---|"]
    for m in METRICS:
        score = "-" if summary[m] is None else f"{summary[m]:.2f}"
        lines.append(f"| {m} | {score} | {summary[f'{m}_n']} |")
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines[2:]))
    print(f"\nSaved: {SCORES.relative_to(ROOT)} (per question), {SUMMARY.relative_to(ROOT)} (table)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--score-only", action="store_true", help="score the saved answers again")
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    if args.score_only:
        rows = json.loads(ANSWERS.read_text(encoding="utf-8"))
    else:
        items = json.loads(EVAL_SET.read_text(encoding="utf-8"))[:args.limit]
        print(f"1) Answering {len(items)} questions with the full system (user: {USER})\n")
        rows = generate_answers(items)
        ANSWERS.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n2) Scoring with RAGAS (judge: {JUDGE_MODEL})\n")
    rows = asyncio.run(score_all(rows[:args.limit]))
    write_report(rows, summarize(rows))


if __name__ == "__main__":
    main()
