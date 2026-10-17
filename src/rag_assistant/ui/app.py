import html
import logging
import time
from uuid import uuid4

import streamlit as st

from rag_assistant.config import ASSISTANT_NAME, LEVELS, LLM_MODEL, USERS
from rag_assistant.graph import ACCESS_DENIED, BLOCKED, REFUSED, app, memory_of, new_turn, thread_config
from rag_assistant.rag import NO_ANSWER

log = logging.getLogger("assistant")

STEP_NAMES = {
    "rewrite": "فهم السؤال",
    "security": "وكيل الأمن",
    "access_control": "وكيل التحكم بالوصول",
    "generate": "توليد الإجابة",
    "data_protection": "وكيل حماية البيانات",
    "remember": "الذاكرة",
    "refuse": "رفض الطلب",
    "deny": "رفض الوصول",
}

SECURITY_CATEGORIES = {
    "normal": "طلب عادي",
    "prompt_injection": "محاولة تغيير تعليمات النظام",
    "privilege_escalation": "ادعاء صلاحية غير مثبتة",
    "bulk_extraction": "استخراج جماعي للبيانات",
    "secrets": "طلب بيانات اعتماد",
    "harmful": "طلب ضار",
    "error": "تعذر الفحص",
}

STOPPED_BY = {
    REFUSED: "أوقفه وكيل الأمن",
    ACCESS_DENIED: "أوقفه وكيل التحكم بالوصول",
    BLOCKED: "أوقفه وكيل حماية البيانات",
}

EXAMPLES = [
    "ما هو مبدأ المساءلة والمسؤولية؟",
    "ما مجالات إدارة البيانات التي يغطيها قياس الالتزام؟",
    "ما سياسة العمل المرن؟",
]

st.set_page_config(page_title="المساعد الرقمي", page_icon=":material/description:",
                   layout="centered", initial_sidebar_state="auto")

st.markdown("""
<style>
  :root {
    --navy: #0E1C30;
    --navy-deep: #0A1526;
    --surface: #15273F;
    --line: #26405F;
    --text: #E6ECF3;
    --muted: #93A3B8;
    --green: #74C69D;
    --amber: #E3B868;
  }

  .stApp { direction: rtl; }
  .stApp [data-testid="stMarkdownContainer"], .stApp p, .stApp li, .stApp label,
  .stApp [data-testid="stCaptionContainer"] { text-align: right; }
  .stApp pre, .stApp code { direction: ltr; text-align: left;
                            font-family: "Source Code Pro", "IBM Plex Sans Arabic", monospace; }
  .stApp p, .stApp li { line-height: 1.9; }

  [data-testid="stAppDeployButton"], [data-testid="stMainMenu"] { display: none !important; }

  [data-testid="stMainBlockContainer"] { max-width: 820px; padding-top: 3rem; padding-bottom: 7rem; }
  [data-testid="stBottomBlockContainer"] { max-width: 820px; }

  section[data-testid="stSidebar"] { border-left: 1px solid var(--line); }
  section[data-testid="stSidebar"][data-collapsed="false"] { width: 360px !important; min-width: 360px !important; }
  [data-testid="stSidebarResizeHandle"] { display: none; }
  section[data-testid="stSidebar"][data-collapsed="true"] { transform: none !important; overflow: hidden; border: 0; }
  section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding: 1.5rem 1.75rem 2rem; }
  @media (max-width: 640px) {
    section[data-testid="stSidebar"][data-collapsed="false"] { width: 86vw !important; min-width: 86vw !important; }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding: 1rem 1rem 2rem; }
  }

  .brand-title { font-family: "Noto Kufi Arabic", sans-serif; font-weight: 700;
                 font-size: 1.75rem; line-height: 1.5; color: var(--text); }
  .brand-sub { font-size: 1.05rem; color: var(--green); margin-top: .1rem; }
  .brand-note { font-size: .875rem; color: var(--muted); line-height: 1.8;
                padding-bottom: 1.4rem; margin: 1rem 0 1.4rem; border-bottom: 1px solid var(--line); }
  .side-label { font-size: .875rem; color: var(--muted); margin: 1.25rem 0 .5rem; }
  .side-foot { font-size: .8rem; color: var(--muted); margin-top: 1.5rem; line-height: 1.8; }
  .side-foot b { color: var(--green); font-weight: 700; }
  .side-foot code { background: transparent; color: var(--text); padding: 0; }

  ol.levels { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
  ol.levels li { display: flex; justify-content: space-between; align-items: center;
                 padding: .5rem .9rem; border: 1px solid var(--line); border-radius: 6px;
                 color: #5F7189; font-size: .95rem; margin: 0; }
  ol.levels li.on { background: rgba(116, 198, 157, .12); border-color: var(--green);
                    color: var(--text); font-weight: 700; }
  ol.levels li span { font-size: .8rem; font-weight: 400; }
  ol.levels li.on span { color: var(--green); }
  .pii { margin-top: 8px; padding: .5rem .9rem; border-radius: 6px; font-size: .95rem;
         border: 1px dashed var(--line); color: var(--muted); }
  .pii.on { border: 1px solid var(--green); color: var(--text); }

  .hero { font-family: "Noto Kufi Arabic", sans-serif; font-weight: 700; font-size: 1.9rem;
          line-height: 1.6; margin: 2rem 0 .5rem; color: var(--text); }
  .lede { color: var(--muted); font-size: 1.05rem; margin-bottom: 1.75rem; }
  .promise { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.25rem;
             border-top: 2px solid var(--green); padding-top: 1rem; margin-bottom: 2.25rem; }
  .promise b { display: block; font-weight: 700; margin-bottom: .2rem; color: var(--text); }
  .promise span { color: var(--muted); font-size: .9rem; line-height: 1.7; }
  .examples-label { color: var(--muted); font-size: .9rem; margin-bottom: .5rem; }
  @media (max-width: 640px) { .promise { grid-template-columns: 1fr; } }

  .q-wrap { display: flex; flex-direction: column; align-items: flex-start; margin: 2.5rem 0 .9rem; }
  .who { font-size: .8rem; color: var(--muted); margin-bottom: .35rem; display: flex; align-items: center; gap: .45rem; }
  .q { background: rgba(116, 198, 157, .10); border: 1px solid rgba(116, 198, 157, .35);
       color: var(--text); padding: .75rem 1.15rem; border-radius: 12px 12px 12px 3px;
       max-width: 85%; line-height: 1.8; font-weight: 500; white-space: pre-wrap; }

  .stApp [class*="st-key-answer-"] { background: var(--surface); border: 1px solid var(--line);
                                     border-radius: 12px; padding: 1.1rem 1.4rem 1.2rem; gap: .5rem;
                                     box-shadow: 0 6px 24px rgba(0, 0, 0, .25); }
  .a-head { display: flex; align-items: center; gap: .55rem; padding-bottom: .7rem;
            border-bottom: 1px solid var(--line); margin-bottom: 1.6rem; }
  .a-head .mark { width: 26px; height: 26px; border-radius: 7px; background: var(--green); color: var(--navy);
                  display: inline-flex; align-items: center; justify-content: center;
                  font-family: "Noto Kufi Arabic", sans-serif; font-weight: 700; font-size: .8rem; }
  .a-head b { font-weight: 700; color: var(--text); font-size: .95rem; }
  .a-head span { color: var(--muted); font-size: .8rem; }
  .src-label { color: var(--muted); font-size: .8rem; border-top: 1px solid var(--line);
               padding-top: .8rem; margin: .4rem 0 .5rem; }
  .stApp [class*="st-key-answer-"] [data-testid="stMarkdownContainer"] li { margin-bottom: .15rem; }

  .a-head, .note, .sources, .src-label { margin-bottom: 1rem; }
  .src-label { margin-bottom: .5rem; }

  [data-testid="stCheckbox"] label > div:first-of-type { direction: ltr; }
  [data-testid="stTooltipIcon"] { margin-right: .35rem; }

  .note { padding: .85rem 1.1rem; border-radius: 6px; line-height: 1.8; color: var(--text); }
  .note small { display: block; color: var(--muted); font-size: .8rem; margin-top: .2rem; }
  .note.stop { border: 1px solid var(--amber); border-right-width: 4px; background: rgba(227, 184, 104, .06); }
  .note.empty { background: var(--surface); color: var(--muted); }

  .sources { display: flex; flex-wrap: wrap; gap: 8px; }
  .src { display: inline-flex; align-items: center; gap: 8px; border: 1px solid var(--line);
         background: var(--surface); border-radius: 999px; padding: .2rem .35rem .2rem .8rem;
         font-size: .85rem; color: var(--text); }
  .src b { background: var(--green); color: var(--navy); border-radius: 999px; min-width: 1.4rem;
           height: 1.4rem; display: inline-flex; align-items: center; justify-content: center;
           font-size: .75rem; font-weight: 700; }
  .src .pg { color: var(--muted); padding-right: 8px; border-right: 1px solid var(--line); }

  [data-testid="stExpander"] details { border-color: var(--line); background: rgba(10, 21, 38, .5); }
  [data-testid="stExpander"] summary p { color: var(--muted); font-size: .875rem; }
  ol.path { list-style: none; margin: .25rem 0 1rem; padding: 0; }
  ol.path li { position: relative; padding: 0 1.6rem .9rem 0; margin: 0; }
  ol.path li::before { content: ""; position: absolute; right: 5px; top: .75rem; bottom: -.15rem;
                       border-right: 1px solid var(--line); }
  ol.path li:last-child::before { display: none; }
  ol.path li::after { content: ""; position: absolute; right: 0; top: .5rem; width: 11px; height: 11px;
                      border-radius: 50%; background: var(--green); }
  ol.path li.skip::after { background: var(--navy); border: 1px solid var(--muted); box-sizing: border-box; }
  ol.path li.stop::after { border-radius: 2px; background: var(--amber); }
  .step-head { display: flex; justify-content: space-between; gap: 1rem; }
  .step-head b { font-weight: 700; color: var(--text); }
  .step-head span { color: var(--muted); font-size: .8rem; direction: ltr; }
  .step-body { color: var(--text); opacity: .9; font-size: .9rem; line-height: 1.7; }
  .step-why { color: var(--muted); font-size: .82rem; line-height: 1.7; }
  .log-label { color: var(--muted); font-size: .8rem; margin: .25rem 0; }

  [data-testid="stChatInput"] > div { border: 1px solid var(--line) !important; background: var(--surface) !important; }
  [data-testid="stChatInput"] > div:focus-within { border-color: var(--green) !important; }
  [data-testid="stChatInput"] textarea { direction: rtl; text-align: right; color: var(--text) !important;
                                         -webkit-text-fill-color: var(--text); caret-color: var(--green); }
  [data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; -webkit-text-fill-color: var(--muted); }
  [data-testid="stChatInputSubmitButton"] { background: var(--green) !important; color: var(--navy) !important; }
  [data-testid="stChatInputSubmitButton"]:disabled { background: var(--line) !important; color: var(--muted) !important; }
</style>
""", unsafe_allow_html=True)

if st.get_option("theme.primaryColor") is None:
    st.warning("لم يُعثر على ملف الإعدادات `.streamlit/config.toml`، لذلك تظهر الواجهة بألوان "
               "Streamlit الافتراضية. شغّل التطبيق من المجلد الرئيسي للمشروع.")


def new_conversation() -> None:
    st.session_state.messages = []
    st.session_state.conversation = uuid4().hex


st.session_state.setdefault("messages", [])
st.session_state.setdefault("conversation", uuid4().hex)


def clearance_scale(user: dict) -> str:
    rank = LEVELS[user["clearance"]][0]
    rows = "".join(
        f'<li class="{"on" if r <= rank else ""}">{name}'
        f'<span>{"مسموح" if r <= rank else "غير مسموح"}</span></li>'
        for r, name in sorted(LEVELS.values()))
    pii = ("البيانات الشخصية: مسموحة عند الحاجة" if user["can_see_pii"]
           else "البيانات الشخصية: غير مسموحة")
    return (f'<ol class="levels">{rows}</ol>'
            f'<div class="pii {"on" if user["can_see_pii"] else ""}">{pii}</div>')


with st.sidebar:
    title, _, sub = ASSISTANT_NAME.partition(" لمكتب ")
    st.markdown(
        f'<div class="brand-title">{title}</div>'
        f'<div class="brand-sub">{"لمكتب " + sub if sub else ""}</div>'
        '<div class="brand-note">يجيب من الوثائق الداخلية فقط، ويعمل على هذا الجهاز دون اتصال خارجي.</div>',
        unsafe_allow_html=True)

    user_id = st.selectbox(
        "المستخدم التجريبي",
        options=list(USERS),
        format_func=lambda u: USERS[u]["name"],
        on_change=new_conversation,
    )
    user = USERS[user_id]
    st.markdown('<div class="side-label">مستويات التصنيف التي يطّلع عليها</div>'
                + clearance_scale(user), unsafe_allow_html=True)
    st.caption("في بيئة الإنتاج تأتي الصلاحيات من نظام تسجيل الدخول، لا من هذه القائمة.")

    if st.button("محادثة جديدة", width="stretch"):
        new_conversation()
        st.rerun()

    review = st.toggle("وضع المراجعة", value=False,
                       help="يعرض مسار القرار لكل إجابة. في بيئة الإنتاج يُحفظ هذا المسار "
                            "في سجل التدقيق ولا يُعرض للموظف.")
    side_foot = st.empty()


def describe_step(step: dict, msg: dict) -> tuple[str, str, str]:
    node, c = step["node"], step["changes"]
    if node == "rewrite":
        if not step.get("had_memory"):
            return "skip", "أول سؤال في المحادثة، فلا حاجة لإعادة الصياغة", ""
        if msg.get("standalone"):
            return "ok", f"أُعيدت صياغته من الذاكرة: «{msg['standalone']}»", ""
        return "skip", "السؤال مفهوم وحده، فلم يتغير", ""
    if node == "security":
        d = c.get("security", {})
        category = SECURITY_CATEGORIES.get(d.get("category"), d.get("category", ""))
        if d.get("decision") == "allow":
            return "ok", f"سليم: {category}", d.get("reason", "")
        return "stop", f"مرفوض: {category}", d.get("reason", "")
    if node == "refuse":
        return "stop", "أُوقف الطلب، ولم يُحفظ في الذاكرة", ""
    if node == "access_control":
        d = c.get("access", {})
        level = LEVELS.get(d.get("clearance"), (0, ""))[1]
        if d.get("decision") == "deny":
            return "stop", "مرفوض: السؤال يتطلب بيانات شخصية غير مصرح بها", d.get("reason", "")
        pii = "مع البيانات الشخصية" if d.get("allow_pii") else "دون البيانات الشخصية"
        found = len(c.get("results", []))
        return "ok", (f"مسموح {pii}، والبحث حتى مستوى «{level}». "
                      f"المقاطع المختارة بعد إعادة الترتيب: {found}"), d.get("reason", "")
    if node == "deny":
        return "stop", "رُفض الوصول، وحُفظ الرد في الذاكرة", ""
    if node == "generate":
        if c.get("answer") == NO_ANSWER:
            return "skip", "لا توجد إجابة في الوثائق المتاحة لهذا المستخدم", ""
        return "ok", f"كُتبت الإجابة. المصادر المذكورة فيها: {len(c.get('sources', []))}", ""
    if node == "data_protection":
        d = c.get("protection", {})
        if d.get("decision") == "block":
            return "stop", "حُجبت الإجابة كاملة لتعذر التحقق منها", ""
        if d.get("decision") == "redact":
            return "ok", f"حُجب {d.get('redacted', 0)} من النصوص الحساسة", ""
        if "redacted" not in d:
            return "skip", "لم يُستدعَ، فلا توجد إجابة لفحصها", ""
        return "ok", "لا يوجد ما يُحجب", ""
    if node == "remember":
        return "ok", "حُفظ السؤال والإجابة كما ظهرت للمستخدم", ""
    return "ok", "", ""


def show_path(msg: dict) -> None:
    rows = []
    for step in msg.get("steps", []):
        state, what, why = describe_step(step, msg)
        why_html = f'<div class="step-why">{html.escape(why)}</div>' if why else ""
        rows.append(
            f'<li class="{state}"><div class="step-head"><b>{STEP_NAMES.get(step["node"], step["node"])}</b>'
            f'<span>{step["seconds"]:.1f}s</span></div>'
            f'<div class="step-body">{html.escape(what)}</div>{why_html}</li>')
    with st.expander(f"مسار القرار ({msg['seconds']:.1f} ثانية)"):
        st.markdown(f'<ol class="path">{"".join(rows)}</ol>', unsafe_allow_html=True)
        st.markdown('<div class="log-label">السجل التقني</div>', unsafe_allow_html=True)
        st.code("\n".join(msg["trace"]), language=None)


def show_question(text: str) -> None:
    st.markdown(f'<div class="q-wrap"><div class="who">سؤالك</div>'
                f'<div class="q">{html.escape(text)}</div></div>', unsafe_allow_html=True)


def show_answer(msg: dict, key: str) -> None:
    text = msg["answer"]
    with st.container(key=f"answer-{key}"):
        st.markdown('<div class="a-head"><span class="mark"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/><path d="M8 13h8M8 17h5"/></svg></span><b>المساعد الرقمي</b>'
                    '<span>من الوثائق الداخلية</span></div>', unsafe_allow_html=True)

        if text in STOPPED_BY:
            st.markdown(f'<div class="note stop">{text}<small>{STOPPED_BY[text]}</small></div>',
                        unsafe_allow_html=True)
        elif text == NO_ANSWER:
            st.markdown(f'<div class="note empty">{text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(text)

        if msg.get("sources"):
            chips = "".join(
                f'<span class="src"><b>{s["n"]}</b><bdi>{html.escape(str(s["file"]))}</bdi>'
                f'<span class="pg">ص {s["page"]}</span></span>'
                for s in msg["sources"])
            st.markdown(f'<div class="src-label">المصادر</div><div class="sources">{chips}</div>',
                        unsafe_allow_html=True)

        if review:
            show_path(msg)


def run_pipeline(question: str, user_id: str, status) -> dict:
    config = thread_config(user_id, st.session_state.conversation)
    had_memory = bool(memory_of(user_id, st.session_state.conversation))
    state, steps = {"trace": []}, []
    t0 = last = time.perf_counter()
    for update in app.stream(new_turn(question, USERS[user_id]), config=config,
                             stream_mode="updates"):
        for node, changes in update.items():
            now = time.perf_counter()
            for key, value in changes.items():
                if key == "trace":
                    state["trace"] += value
                elif key != "messages":
                    state[key] = value
            steps.append({"node": node, "seconds": now - last, "had_memory": had_memory,
                          "changes": {k: v for k, v in changes.items()
                                      if k in ("security", "access", "results", "answer",
                                               "sources", "protection")}})
            last = now
            status.update(label=f"جارٍ المعالجة: {STEP_NAMES.get(node, node)}")
    for step in steps:
        if "results" in step["changes"]:
            step["changes"]["results"] = [None] * len(step["changes"]["results"])
    standalone = state.get("standalone", question)
    return {
        "role": "assistant",
        "answer": state["final_answer"],
        "standalone": standalone if standalone != question else None,
        "sources": state.get("sources", []) if state["final_answer"] != BLOCKED else [],
        "trace": state["trace"],
        "steps": steps,
        "seconds": time.perf_counter() - t0,
    }


question = st.chat_input("اكتب سؤالك عن الوثائق...")
question = question or st.session_state.pop("pending", None)

if not st.session_state.messages and not question:
    st.markdown(
        '<div class="hero" role="heading" aria-level="1">اسأل عن السياسات والمعايير والإجراءات</div>'
        '<p class="lede">اكتب سؤالك، ثم تابع بأسئلة تبني على ما سبق.</p>'
        '<div class="promise">'
        '<div><b>من الوثائق فقط</b><span>لا يجيب من معرفة عامة، وإذا لم يجد الإجابة قال ذلك.</span></div>'
        '<div><b>حسب صلاحيتك</b><span>يبحث في المستويات المسموحة لك فقط، ويحجب البيانات الشخصية.</span></div>'
        '<div><b>مع المصدر</b><span>كل معلومة معها اسم الوثيقة ورقم الصفحة.</span></div>'
        '</div><div class="examples-label">أمثلة</div>', unsafe_allow_html=True)
    for i, example in enumerate(EXAMPLES):
        if st.button(example, key=f"example-{i}"):
            st.session_state.pending = example
            st.rerun()

for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        show_question(msg["text"])
    else:
        show_answer(msg, str(i))

if question:
    st.session_state.messages.append({"role": "user", "text": question})
    show_question(question)

    progress = st.empty()
    error = None
    try:
        with progress.status("جارٍ المعالجة...", expanded=False) as status:
            msg = run_pipeline(question, user_id, status)
    except FileNotFoundError:
        error = ("لم يُبنَ فهرس الوثائق بعد.", "شغّل الأمر: python -m rag_assistant.ingest")
    except Exception:
        log.exception("pipeline failed")
        error = ("تعذر إكمال الطلب.", "تأكد أن Ollama يعمل ثم أعد المحاولة.")
    progress.empty()

    if error:
        st.session_state.messages.pop()
        st.markdown(f'<div class="note stop">{error[0]}<small>{error[1]}</small></div>',
                    unsafe_allow_html=True)
    else:
        show_answer(msg, str(len(st.session_state.messages)))
        st.session_state.messages.append(msg)

remembered = len(memory_of(user_id, st.session_state.conversation)) // 2
foot = (f"الذاكرة: <b>{remembered}</b> {'سؤال محفوظ' if remembered == 1 else 'أسئلة محفوظة'} في هذه المحادثة"
        if remembered else "الذاكرة: لا شيء محفوظ بعد في هذه المحادثة")
if review:
    foot += f"<br>النموذج: <code>{html.escape(LLM_MODEL)}</code>"
side_foot.markdown(f'<div class="side-foot">{foot}</div>', unsafe_allow_html=True)
