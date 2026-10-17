import pytest
from langchain_core.documents import Document

from rag_assistant import rerank as rerank_module
from rag_assistant.rag import NO_ANSWER, cited_sources, generate, keep_relevant
from rag_assistant.rerank import rerank


def chunk(text="نص", source="a.pdf", page=1):
    return Document(page_content=text, metadata={"source": source, "page": page})


class LengthScorer:
    def predict(self, pairs):
        return [float(len(text)) for _, text in pairs]


def test_citations_map_to_file_and_page():
    results = [(chunk(source="a.pdf", page=3), 0.71), (chunk(source="b.pdf", page=8), 0.66)]
    sources, invalid = cited_sources("الإجابة [2] وأيضًا [1][2] و[9].", results)
    assert [(s["n"], s["file"], s["page"]) for s in sources] == [(1, "a.pdf", 3), (2, "b.pdf", 8)]
    assert invalid == [9]


def test_chunks_below_the_similarity_threshold_are_dropped():
    results = [(chunk(), 0.62), (chunk(), 0.38)]
    assert keep_relevant(results) == results[:1]


def test_no_chunks_gives_the_fixed_answer_without_calling_the_model():
    assert generate("كم مدة الإجازة السنوية؟", []) == (NO_ANSWER, {})


def test_reranker_reorders_but_never_adds_chunks(monkeypatch):
    monkeypatch.setattr(rerank_module, "get_reranker", lambda: LengthScorer())
    results = [(chunk("قصير"), 0.9), (chunk("نص متوسط الطول"), 0.8), (chunk("أطول نص في هذه القائمة كلها"), 0.7)]
    kept, note = rerank("سؤال", results, top_n=2)
    assert [doc.page_content for doc, _ in kept] == ["أطول نص في هذه القائمة كلها", "نص متوسط الطول"]
    assert {id(doc) for doc, _ in kept} <= {id(doc) for doc, _ in results}
    assert "1 brought up" in note
    assert kept[0][0].metadata["rerank_score"] == pytest.approx(len("أطول نص في هذه القائمة كلها"))


def test_search_order_is_kept_when_the_reranker_is_missing(monkeypatch):
    monkeypatch.setattr(rerank_module, "get_reranker", lambda: None)
    results = [(chunk("أ"), 0.9), (chunk("ب"), 0.8), (chunk("ج"), 0.7)]
    kept, note = rerank("سؤال", results, top_n=2)
    assert kept == results[:2]
    assert note == "model not available, search order kept"
