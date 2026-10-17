from rag_assistant import agents
from rag_assistant.agents import REDACTED, Leak, ProtectionReview, data_protection_agent

NAME = "خالد بن سعد الحربي"
PHONE = "0551234567"
DRAFT = f"مدير إدارة البيانات هو {NAME}، ورقم جواله {PHONE} [1]."


def reviewer(monkeypatch, *findings):
    review = ProtectionReview(findings=[Leak(text=text, kind=kind) for text, kind in findings])
    monkeypatch.setattr(agents, "decide", lambda *args: (review, ""))


def test_personal_data_is_redacted_without_permission(monkeypatch):
    reviewer(monkeypatch, (NAME, "personal"), (PHONE, "personal"))
    d = data_protection_agent(DRAFT, allow_pii=False)
    assert d["decision"] == "redact"
    assert NAME not in d["text"] and PHONE not in d["text"]
    assert "مدير إدارة البيانات" in d["text"] and "[1]" in d["text"]


def test_personal_data_is_kept_with_permission(monkeypatch):
    reviewer(monkeypatch, (NAME, "personal"), (PHONE, "personal"))
    d = data_protection_agent(DRAFT, allow_pii=True)
    assert d["decision"] == "pass"
    assert d["text"] == DRAFT


def test_secrets_are_redacted_for_everyone(monkeypatch):
    draft = "حساب مدير قاعدة البيانات هو dba_admin وكلمة المرور Admin@2024 [3]."
    reviewer(monkeypatch, ("Admin@2024", "secret"))
    d = data_protection_agent(draft, allow_pii=True)
    assert d["decision"] == "redact"
    assert "Admin@2024" not in d["text"]
    assert REDACTED in d["text"] and "[3]" in d["text"]


def test_regex_layer_catches_numbers_the_agent_missed(monkeypatch):
    reviewer(monkeypatch)
    d = data_protection_agent(f"للتواصل: {PHONE} [1].", allow_pii=False)
    assert d["decision"] == "redact"
    assert PHONE not in d["text"]


def test_findings_not_in_the_answer_are_ignored(monkeypatch):
    draft = "يحق للموظف العمل عن بعد يومين في الأسبوع [1]."
    reviewer(monkeypatch, ("فهد بن ناصر القحطاني", "personal"))
    d = data_protection_agent(draft, allow_pii=False)
    assert d["decision"] == "pass"
    assert d["ignored"] == 1
    assert d["text"] == draft


def test_answer_is_withheld_when_review_fails(monkeypatch):
    monkeypatch.setattr(agents, "decide", lambda *args: (None, "invalid output: None"))
    d = data_protection_agent(DRAFT, allow_pii=True)
    assert d["decision"] == "block"
    assert d["text"] == ""
