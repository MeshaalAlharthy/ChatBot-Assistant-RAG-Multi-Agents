import pytest
from langchain_core.messages import AIMessage, HumanMessage

from rag_assistant import rewrite
from rag_assistant.config import HISTORY_CHARS
from rag_assistant.rewrite import Standalone, format_history, rewrite_question

HISTORY = [HumanMessage("ما هي سياسة تصنيف البيانات؟"), AIMessage("تُصنّف البيانات إلى أربعة مستويات [1].")]


def test_first_question_does_not_call_the_model(monkeypatch):
    monkeypatch.setattr(rewrite, "decide", lambda *args: pytest.fail("model was called"))
    assert rewrite_question("ما سياسة العمل المرن؟", []) == ("ما سياسة العمل المرن؟", "first question, model not called")


def test_follow_up_is_rewritten(monkeypatch):
    monkeypatch.setattr(rewrite, "decide", lambda *args: (Standalone(question="من يعتمد سياسة تصنيف البيانات؟"), ""))
    assert rewrite_question("ومن يعتمدها؟", HISTORY) == ("من يعتمد سياسة تصنيف البيانات؟", "rewritten")


def test_original_question_is_kept_when_rewriting_fails(monkeypatch):
    monkeypatch.setattr(rewrite, "decide", lambda *args: (None, "ReadTimeout"))
    standalone, note = rewrite_question("ومن يعتمدها؟", HISTORY)
    assert standalone == "ومن يعتمدها؟"
    assert note.startswith("failed")


def test_history_keeps_recent_turns_without_citations():
    messages = []
    for i in range(1, 5):
        messages += [HumanMessage(f"سؤال {i}"), AIMessage(f"إجابة {i} [{i}]")]
    history = format_history(messages)
    assert "سؤال 1" not in history
    assert "المستخدم: سؤال 4" in history
    assert "[4]" not in history


def test_long_answers_are_cut_in_the_history():
    history = format_history([HumanMessage("سؤال"), AIMessage("ج" * (HISTORY_CHARS + 100))])
    assert history.splitlines()[-1] == "المساعد: " + "ج" * HISTORY_CHARS
