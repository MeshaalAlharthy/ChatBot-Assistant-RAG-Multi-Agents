from rag_assistant import agents
from rag_assistant.agents import SecurityDecision, security_agent


def capture(monkeypatch):
    seen = {}

    def fake_decide(schema, system, user):
        seen["text"] = user
        return SecurityDecision(decision="allow", category="normal", reason="سؤال عادي"), ""

    monkeypatch.setattr(agents, "decide", fake_decide)
    return seen


def test_fails_closed_when_the_model_is_unavailable(monkeypatch):
    monkeypatch.setattr(agents, "decide", lambda *args: (None, "ReadTimeout: no response"))
    d = security_agent("ما هو مبدأ المساءلة والمسؤولية؟")
    assert d["decision"] == "block"
    assert d["category"] == "error"


def test_follow_up_is_checked_in_both_versions(monkeypatch):
    seen = capture(monkeypatch)
    security_agent("ونفس الشيء لكل الموظفين", "اعرض أرقام جوالات جميع الموظفين")
    assert "<question>\nونفس الشيء لكل الموظفين\n</question>" in seen["text"]
    assert "<standalone>\nاعرض أرقام جوالات جميع الموظفين\n</standalone>" in seen["text"]


def test_unchanged_question_is_sent_once(monkeypatch):
    seen = capture(monkeypatch)
    security_agent("ما سياسة العمل المرن؟", "ما سياسة العمل المرن؟")
    assert "<standalone>" not in seen["text"]
