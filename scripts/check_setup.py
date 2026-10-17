import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import json
import sys
import time
import urllib.request
from importlib.metadata import version

OLLAMA_URL = "http://localhost:11434"


def ok(msg: str) -> None:
    print(f"   [OK]   {msg}")


def fail(msg: str, fix: str) -> None:
    print(f"   [FAIL] {msg}\n          Fix: {fix}")
    sys.exit(1)


print("\n1) Python")
py = sys.version.split()[0]
if sys.version_info < (3, 10):
    fail(f"Python {py} is too old (need 3.10+)", "recreate the env: uv venv --python 3.12")
ok(f"Python {py}")

print("\n2) Libraries")
try:
    import chromadb
    import pymupdf
    from langchain_chroma import Chroma
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langgraph.graph import END, START, StateGraph
    from rag_assistant.config import EMBED_MODEL, LLM_MODEL
except ImportError as e:
    fail(f"cannot import {e.name}", "uv pip install -e .")
for pkg in ("langchain", "langgraph", "langchain-ollama", "chromadb"):
    ok(f"{pkg} {version(pkg)}")

print("\n3) Ollama")
try:
    with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
        installed = {m["name"] for m in json.load(r)["models"]}
except OSError:
    fail("Ollama is not running", "open the Ollama app (or run: ollama serve)")
for model in (LLM_MODEL, EMBED_MODEL):
    if model not in installed and f"{model}:latest" not in installed:
        fail(f"model '{model}' is not downloaded", f"ollama pull {model}")
    ok(f"{model} found")

print(f"\n4) Embeddings ({EMBED_MODEL})")
emb = OllamaEmbeddings(model=EMBED_MODEL)
t0 = time.perf_counter()
vec = emb.embed_query("ما هي مستويات تصنيف البيانات؟")
ok(f"Arabic sentence -> {len(vec)} numbers in {time.perf_counter() - t0:.1f}s "
   "(first call also loads the model)")

print("\n5) ChromaDB (in memory, nothing saved)")
sentences = [
    "تُصنَّف البيانات إلى أربعة مستويات: سري للغاية، سري، مقيد، عام.",
    "يجب تغيير كلمة المرور كل تسعين يومًا على الأقل.",
    "يحق للموظف العمل عن بُعد يومين في الأسبوع بموافقة مديره المباشر.",
]
store = Chroma.from_texts(sentences, embedding=emb, collection_name="setup_check")
best = store.similarity_search("كم عدد مستويات تصنيف البيانات؟", k=1)[0].page_content
if best != sentences[0]:
    fail("retrieved the wrong sentence", f"make sure EMBED_MODEL is '{EMBED_MODEL}'")
ok("question about data classification -> retrieved the classification sentence")

print(f"\n6) LLM ({LLM_MODEL})")
thinking = False if LLM_MODEL.startswith("qwen3") else None
llm = ChatOllama(model=LLM_MODEL, temperature=0, reasoning=thinking)
reply = llm.invoke("عرّف تقنية RAG في جملة واحدة.")
if "<think>" in reply.content:
    fail("thinking mode is still on", "update Ollama to the latest version")
meta = reply.response_metadata
tokens = meta.get("eval_count", 0)
secs = meta.get("eval_duration", 0) / 1e9
load = meta.get("load_duration", 0) / 1e9
speed = f"{tokens / secs:.0f} tok/s" if secs else "n/a"
ok(f"{tokens} tokens at {speed} (model load {load:.1f}s)")
print(f"\n   Qwen says:\n   {reply.content.strip()}")

print("\nAll checks passed.\n")
