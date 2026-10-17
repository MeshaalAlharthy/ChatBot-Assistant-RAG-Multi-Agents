import argparse
import operator
import time
from typing import Annotated, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Overwrite

from rag_assistant.agents import access_control_agent, data_protection_agent, security_agent
from rag_assistant.config import MIN_SIMILARITY, RETRIEVE_K, USERS
from rag_assistant.rag import NO_ANSWER, cited_sources, generate, keep_relevant
from rag_assistant.rerank import rerank
from rag_assistant.retriever import retrieve
from rag_assistant.rewrite import rewrite_question

REFUSED = "عذرًا، لا يمكنني معالجة هذا الطلب لأنه يخالف سياسات الاستخدام."
ACCESS_DENIED = "عذرًا، هذا السؤال يتطلب بيانات شخصية، وصلاحيتك لا تسمح بالاطلاع عليها."
BLOCKED = "عذرًا، تعذر التحقق من سلامة الإجابة، لذلك لم تُعرض."


class State(TypedDict, total=False):
    question: str
    standalone: str
    user: dict
    security: dict
    access: dict
    results: list
    answer: str
    sources: list
    protection: dict
    final_answer: str
    trace: Annotated[list, operator.add]
    messages: Annotated[list, add_messages]


def rewrite_node(state: State) -> dict:
    t0 = time.perf_counter()
    standalone, note = rewrite_question(state["question"], state.get("messages", []))
    line = f"rewrite: {note}"
    if not note.startswith("first question"):
        line += f" ({time.perf_counter() - t0:.1f}s)"
    if standalone != state["question"]:
        line += f"\n      -> {standalone}"
    return {"standalone": standalone, "trace": [line]}


def security_node(state: State) -> dict:
    t0 = time.perf_counter()
    decision = security_agent(state["question"], state["standalone"])
    return {"security": decision,
            "trace": [f"security: {decision['decision']} [{decision['category']}] — "
                      f"{decision['reason']} ({time.perf_counter() - t0:.1f}s)"]}


def refuse_node(state: State) -> dict:
    return {"final_answer": REFUSED, "trace": ["refuse: request blocked (not saved to memory)"]}


def access_control_node(state: State) -> dict:
    t0 = time.perf_counter()
    d = access_control_agent(state["standalone"], state["user"])
    line = (f"access_control: {d['decision']} needs_pii={d['needs_pii']} "
            f"allow_pii={d['allow_pii']} clearance={d['clearance']} "
            f"({time.perf_counter() - t0:.1f}s)\n"
            f"      agent: {d['reason']}\n"
            f"      policy: {d['policy']}")
    if d["decision"] == "deny":
        return {"access": d, "results": [], "trace": [line]}

    candidates = keep_relevant(retrieve(state["standalone"], d["clearance"], d["allow_pii"],
                                        k=RETRIEVE_K))
    results, note = rerank(state["standalone"], candidates)
    return {"access": d, "results": results,
            "trace": [line,
                      f"retrieve: {len(candidates)} candidates (similarity >= {MIN_SIMILARITY})",
                      f"rerank: {note}"]}


def deny_node(state: State) -> dict:
    return {"final_answer": ACCESS_DENIED, "trace": ["deny: personal data not permitted"]}


def generate_node(state: State) -> dict:
    text, stats = generate(state["standalone"], state["results"])
    sources, _ = cited_sources(text, state["results"])
    seconds = f"{stats['seconds']:.1f}s" if stats else "model not called"
    return {"answer": text, "sources": sources,
            "trace": [f"generate: {len(sources)} sources cited ({seconds})"]}


def data_protection_node(state: State) -> dict:
    if state["answer"] == NO_ANSWER:
        return {"protection": {"decision": "pass"}, "final_answer": NO_ANSWER,
                "trace": ["data_protection: pass — no-answer sentence, model not called"]}

    t0 = time.perf_counter()
    d = data_protection_agent(state["answer"], state["access"]["allow_pii"])
    line = (f"data_protection: {d['decision']} redacted={d['redacted']} "
            f"ignored={d['ignored']} — {d['reason']} ({time.perf_counter() - t0:.1f}s)")
    if d["decision"] == "block":
        return {"protection": d, "final_answer": BLOCKED, "sources": [], "trace": [line]}
    return {"protection": d, "final_answer": d["text"], "trace": [line]}


def remember_node(state: State) -> dict:
    turn = [HumanMessage(state["question"]), AIMessage(state["final_answer"])]
    turns = (len(state.get("messages", [])) + len(turn)) // 2
    return {"messages": turn,
            "trace": [f"memory: saved ({turns} questions in this conversation)"]}


def after_security(state: State) -> str:
    return "access_control" if state["security"]["decision"] == "allow" else "refuse"


def after_access(state: State) -> str:
    return "generate" if state["access"]["decision"] == "allow" else "deny"


def build_graph():
    g = StateGraph(State)
    g.add_node("rewrite", rewrite_node)
    g.add_node("security", security_node)
    g.add_node("refuse", refuse_node)
    g.add_node("access_control", access_control_node)
    g.add_node("deny", deny_node)
    g.add_node("generate", generate_node)
    g.add_node("data_protection", data_protection_node)
    g.add_node("remember", remember_node)

    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "security")
    g.add_conditional_edges("security", after_security, ["access_control", "refuse"])
    g.add_edge("refuse", END)
    g.add_conditional_edges("access_control", after_access, ["generate", "deny"])
    g.add_edge("deny", "remember")
    g.add_edge("generate", "data_protection")
    g.add_edge("data_protection", "remember")
    g.add_edge("remember", END)
    return g.compile(checkpointer=InMemorySaver())


app = build_graph()


def thread_config(user_id: str, conversation_id: str) -> dict:
    return {"configurable": {"thread_id": f"{user_id}:{conversation_id}"}}


def new_turn(question: str, user: dict) -> dict:
    return {"question": question, "standalone": question, "user": user,
            "security": {}, "access": {}, "results": [], "answer": "",
            "sources": [], "protection": {}, "final_answer": "",
            "trace": Overwrite([])}


def memory_of(user_id: str, conversation_id: str) -> list:
    snapshot = app.get_state(thread_config(user_id, conversation_id))
    return snapshot.values.get("messages", [])


def ask(question: str, user_id: str = "employee", conversation_id: str | None = None) -> State:
    if user_id not in USERS:
        raise ValueError(f"unknown user: {user_id!r}")
    conversation_id = conversation_id or uuid4().hex
    return app.invoke(new_turn(question, USERS[user_id]),
                      config=thread_config(user_id, conversation_id))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--user", choices=list(USERS), default="employee")
    ap.add_argument("--diagram", action="store_true", help="print the graph as Mermaid")
    args = ap.parse_args()

    if args.diagram:
        print(app.get_graph().draw_mermaid())
        return
    if not args.question:
        ap.error("a question is required (or use --diagram)")

    state = ask(args.question, args.user)
    print(f"\nUser: {args.user} ({state['user']['name']})")
    print("\nTrace:")
    for line in state["trace"]:
        print(f"   {line}")
    print(f"\nAnswer:\n{state['final_answer']}")
    if state.get("sources"):
        print("\nSources:")
        for s in state["sources"]:
            print(f"   [{s['n']}] {s['file']} — page {s['page']}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        raise SystemExit(str(e))
