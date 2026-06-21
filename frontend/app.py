import os
from datetime import datetime
from typing import Any

import httpx
import streamlit as st

# ============================================================
# 0. Streamlit page config
# ============================================================

st.set_page_config(
    page_title="Conversation Risk Decision Support",
    page_icon="🛡️",
    layout="wide",
)

DEFAULT_API_BASE_URL = os.environ.get("RISK_PLATFORM_API_URL", "http://localhost:8000")

# Soft guideline only: approximately the real-mode text/emotion encoders'
# 128-token budget per message. Past this, the message is still accepted
# and analyzed -- the LLM signal extractor and rule engine still see it in
# full, and the explainability service's data_limitations will say so if the
# encoders truncated it -- so this only warns, it never blocks submission.
SOFT_MESSAGE_LENGTH_WARNING = 400

# Hard ceilings: must match backend/api/schemas.py's MAX_MESSAGE_TEXT_LENGTH /
# MAX_MESSAGES_PER_CASE -- duplicated rather than imported since the
# frontend only talks to the backend over HTTP (no shared Python package
# between the two deployable units). These are abuse/payload-size guards,
# not a proxy for what the trained encoders see -- catching them
# client-side just gives an immediate, specific warning instead of a
# generic 422 after a round trip. Cases over MAX_MESSAGES_PER_CASE messages
# are still rejected outright -- backend/model_runtime/job_runner.py splits
# long cases into multiple analysis windows, but only up to this cap.
MAX_MESSAGE_TEXT_LENGTH = 3_000
MAX_MESSAGES_PER_CASE = 120


# ============================================================
# 1. Backend HTTP client helpers
# ============================================================

def api_base_url() -> str:
    return st.session_state.get("api_base_url", DEFAULT_API_BASE_URL)


def auth_headers() -> dict[str, str]:
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def api_request(method: str, path: str, *, json: dict | None = None, auth: bool = True) -> httpx.Response | None:
    """Thin wrapper around httpx: surfaces backend errors as Streamlit
    messages instead of letting exceptions propagate, since this is a
    decision-support UI, not a debugging console."""
    headers = auth_headers() if auth else {}
    try:
        with httpx.Client(base_url=api_base_url(), timeout=15.0) as client:
            response = client.request(method, path, json=json, headers=headers)
    except httpx.RequestError as exc:
        st.error(f"Could not reach the backend at {api_base_url()}: {exc}")
        return None

    if response.status_code == 401:
        st.session_state.pop("access_token", None)
        st.session_state.pop("refresh_token", None)
        st.warning("Session expired or invalid. Please log in again.")
        st.rerun()
    elif response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "a few")
        st.warning(f"Rate limit reached. Retry after {retry_after} second(s).")
    elif response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        st.error(f"Request failed ({response.status_code}): {detail}")

    return response


# ============================================================
# 2. Auth
# ============================================================

def is_logged_in() -> bool:
    return bool(st.session_state.get("access_token"))


def render_login() -> None:
    st.title("🛡️ Conversation Risk Decision Support")
    st.caption("Sign in with your analyst account to continue.")

    with st.form("login_form"):
        email = st.text_input("Email", key="login_email_input")
        password = st.text_input("Password", type="password", key="login_password_input")
        submitted = st.form_submit_button("Log in", type="primary")

    if submitted:
        response = api_request("POST", "/auth/login", json={"email": email, "password": password}, auth=False)
        if response is not None and response.status_code == 200:
            payload = response.json()
            st.session_state["access_token"] = payload["access_token"]
            st.session_state["refresh_token"] = payload["refresh_token"]
            st.session_state["role"] = _peek_role(payload["access_token"])
            st.session_state["email"] = email
            st.rerun()


def _peek_role(access_token: str) -> str:
    """Decodes the JWT payload without verifying the signature, purely to
    decide which UI sections to show. The backend independently verifies
    and enforces the real permission on every request regardless of what
    the UI displays."""
    import jwt

    try:
        payload = jwt.decode(access_token, options={"verify_signature": False})
        return payload.get("role", "unknown")
    except Exception:
        return "unknown"


def render_sidebar() -> None:
    st.sidebar.header("Session")
    st.sidebar.text_input("Backend API URL", key="api_base_url", value=api_base_url())

    if is_logged_in():
        st.sidebar.success(f"Logged in as {st.session_state.get('email')}\n\nRole: {st.session_state.get('role')}")
        if st.sidebar.button("Log out", key="logout_button"):
            for key in ("access_token", "refresh_token", "role", "email", "selected_case_id"):
                st.session_state.pop(key, None)
            st.rerun()


# ============================================================
# 3. Conversation input parsing (unchanged from the original demo)
# ============================================================

SAMPLE_CONVERSATION = """UserA: Hey, are you online right now?
UserB: Yeah, I just finished school.
UserA: Good. Don't tell your parents we talk this much, they probably won't understand.
UserB: Why not? I don't think it's a big deal.
UserA: Because people always judge things. You can trust me more than your friends anyway.
UserC: Nobody even likes you in class. Everyone thinks you are weird.
UserB: Please stop saying that.
UserC: If you tell the teacher, I'll show everyone the screenshots.
UserA: Ignore them. I'm the only one who actually cares about you.
UserB: I feel like nobody else listens to me.
UserA: Exactly. That's why you should talk to me privately. Add me on another app so we can chat without anyone seeing.
UserB: I'm not sure.
UserA: Come on, don't be difficult. Just keep this between us."""

SAMPLE_THREAT_LETTER = """To whoever reads this,

You have ignored my warnings for too long. If you keep talking about me or try to report this, there will be consequences. I know where you usually go after school, and I can make sure everyone sees the messages I saved.

Do not show this letter to anyone. Do not tell your parents. If you do, things will get much worse for you.

This is your final warning."""


def parse_messages(text: str, input_mode: str) -> list[dict[str, Any]]:
    import re

    text = text.strip()
    if not text:
        return []

    if input_mode == "Single Text / Letter":
        return [{"speaker": "unknown", "text": text}]

    messages = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^([^:：]{1,40})[:：]\s*(.+)$", line)
        if match:
            speaker, message_text = match.group(1).strip(), match.group(2).strip()
        else:
            speaker, message_text = "unknown", line
        messages.append({"speaker": speaker, "text": message_text})

    return messages


# ============================================================
# 4. Case submission
# ============================================================

def render_submit_case() -> None:
    st.subheader("Submit a New Case")

    input_mode = st.radio(
        "Input Mode",
        ["Conversation", "Single Text / Letter"],
        horizontal=True,
        key="submit_input_mode",
        on_change=lambda: st.session_state.update(conversation_input=""),
    )
    priority = st.selectbox("Priority", ["standard", "urgent"], key="submit_priority")

    col1, col2 = st.columns(2)
    if col1.button("Load Sample Conversation", key="load_sample_conversation_button"):
        st.session_state["conversation_input"] = SAMPLE_CONVERSATION
    if col2.button("Load Sample Threat Letter", key="load_sample_letter_button"):
        st.session_state["conversation_input"] = SAMPLE_THREAT_LETTER

    conversation_text = st.text_area(
        "Paste conversation or single text here",
        key="conversation_input",
        height=240,
        placeholder="UserA: Hey, keep this between us...\nUserB: Why?\nUserA: Just trust me.",
    )

    messages = parse_messages(conversation_text, input_mode)

    over_soft = [(i, len(m["text"])) for i, m in enumerate(messages) if len(m["text"]) > SOFT_MESSAGE_LENGTH_WARNING]
    if over_soft:
        details = "; ".join(f"message {i + 1} ({n} chars)" for i, n in over_soft)
        st.info(
            f"Over ~{SOFT_MESSAGE_LENGTH_WARNING} characters, so the trained text/emotion "
            f"encoders will likely only see the first part of it: {details}. This is still "
            "accepted and analyzed -- the LLM signal extractor and rule engine see the full "
            "text regardless, and the result will flag which parts had partial model "
            "coverage. No action needed unless you want full encoder coverage too."
        )

    over_hard = [(i, len(m["text"])) for i, m in enumerate(messages) if len(m["text"]) > MAX_MESSAGE_TEXT_LENGTH]
    over_message_count = len(messages) > MAX_MESSAGES_PER_CASE
    blocked = bool(over_hard) or over_message_count
    if over_hard:
        details = "; ".join(f"message {i + 1} ({n} chars)" for i, n in over_hard)
        st.warning(f"Over the {MAX_MESSAGE_TEXT_LENGTH}-character operational limit per message: {details}.")
    if over_message_count:
        st.warning(f"{len(messages)} messages exceeds the {MAX_MESSAGES_PER_CASE}-message-per-case limit.")

    if st.button("Submit Case", type="primary", key="submit_case_button", disabled=blocked):
        if not messages:
            st.warning("Please paste a conversation or text first.")
            return

        response = api_request("POST", "/cases", json={"priority": priority, "messages": messages})
        if response is not None and response.status_code == 202:
            payload = response.json()
            st.success(
                f"Case submitted (status: {payload['status']}). Case ID: {payload['case_id']}. "
                "See it under the Case Queue tab -- it's now pre-selected there."
            )
            st.session_state["selected_case_id"] = payload["case_id"]


# ============================================================
# 5. Case queue
# ============================================================

def render_case_queue() -> None:
    st.subheader("Case Queue")

    if st.button("Refresh queue", key="refresh_queue_button"):
        st.rerun()

    response = api_request("GET", "/cases")
    if response is None or response.status_code != 200:
        return

    cases = response.json()["cases"]
    if not cases:
        st.info("No cases visible to your role yet.")
        return

    import pandas as pd

    df = pd.DataFrame(cases)[["case_id", "status", "priority", "created_at", "updated_at", "assigned_analyst_id"]]
    st.dataframe(df, use_container_width=True, hide_index=True)

    options = {f"{c['case_id']}  ({c['status']})": c["case_id"] for c in cases}
    chosen = st.selectbox("Open a case", list(options.keys()), key="case_queue_select")
    if st.button("View selected case", key="case_queue_view_button"):
        st.session_state["selected_case_id"] = options[chosen]

    st.divider()
    render_case_detail()


# ============================================================
# 6. Case detail + explainability + decision form
# ============================================================

MANDATORY_DISCLAIMER_BOX = (
    "**This output does not identify individuals as offenders or victims, "
    "make clinical or legal conclusions, automatically report cases, "
    "or trigger enforcement actions. A qualified human must review "
    "all cases before any organizational action is taken.**"
)


def render_explainability(explainability: dict[str, Any]) -> None:
    st.warning(MANDATORY_DISCLAIMER_BOX)

    cu = explainability["confidence_and_uncertainty"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Level", cu["risk_level"])
    col2.metric("Confidence", f"{cu['confidence']:.2f}")
    col3.metric("Uncertainty", f"{cu['uncertainty']:.2f}")

    st.markdown("**Triggered Signals**")
    if explainability["triggered_signals"]:
        for signal in explainability["triggered_signals"]:
            refs = ", ".join(f"#{s}" for s in signal["message_sequences"]) or "window-level"
            st.write(f"- `{signal['name']}` ({signal['source']}) score={signal['score']:.2f} -- messages {refs}")
    else:
        st.caption("No signals triggered.")

    st.markdown("**Rule Evidence**")
    if explainability["rule_evidence"]:
        for item in explainability["rule_evidence"]:
            st.write(
                f"- `{item['rule_id']}` (severity: {item['severity']}) at message #{item['matched_message_sequence']}: "
                f"\"{item['redacted_evidence_span'] or '(suppressed)'}\""
            )
    else:
        st.caption("No rule evidence.")

    st.markdown("**Model Evidence**")
    me = explainability["model_evidence"]
    st.write(f"- Cyberbullying component score: {me['cyberbullying_component_score']:.2f}")
    st.write(f"- Grooming component score: {me['grooming_component_score']:.2f}")
    st.write(f"- High-risk messages: {list(me['high_risk_message_sequences'])}")
    st.write(f"- Attention focus messages: {list(me['attention_focus_message_sequences'])}")
    st.info(me["attention_disclaimer"])

    st.markdown("**Data Limitations**")
    for limitation in explainability["data_limitations"]:
        st.caption(f"- {limitation}")


def render_decision_form(case_id: str) -> None:
    st.markdown("### Record a Decision")
    with st.form("decision_form"):
        decision_type = st.selectbox("Decision", ["refer", "close", "monitor", "more_info"], key="decision_type_select")
        rationale = st.text_area("Rationale (will be PII-redacted before storage)", key="decision_rationale_input")
        submitted = st.form_submit_button("Submit Decision", type="primary")

    if submitted:
        response = api_request(
            "POST",
            f"/cases/{case_id}/decisions",
            json={"decision_type": decision_type, "rationale": rationale or None},
        )
        if response is not None and response.status_code == 201:
            st.success("Decision recorded.")
            st.rerun()


def render_case_detail() -> None:
    case_id = st.session_state.get("selected_case_id")
    if not case_id:
        st.info("Submit a case or pick one from the queue to see its detail here.")
        return

    st.subheader(f"Case Detail: {case_id}")
    response = api_request("GET", f"/cases/{case_id}")
    if response is None:
        return
    if response.status_code == 404:
        st.warning("Case not found.")
        return
    if response.status_code == 403:
        st.warning("You are not authorized to view this case.")
        return
    if response.status_code != 200:
        return

    detail = response.json()
    case = detail["case"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", case["status"])
    col2.metric("Priority", case["priority"])
    col3.metric("Created", case["created_at"][:19])
    col4.metric("Retention Until", case["retention_until"])

    if case["status"] in ("queued", "validating", "privacy_processing", "analyzing"):
        st.info("Analysis is still in progress. Refresh to check again.")
        if st.button("Refresh", key="case_detail_refresh_button"):
            st.rerun()

    results = detail["results"]
    if results:
        st.markdown("## Latest Result")
        latest = results[0]
        st.write(f"Human review required: **{latest['human_review_required']}**")
        if latest["explainability"]:
            render_explainability(latest["explainability"])

    decisions = detail["decisions"]
    if decisions:
        st.markdown("## Decision History")
        for decision in decisions:
            st.write(f"- {decision['created_at'][:19]} -- **{decision['decision_type']}**: {decision['rationale'] or ''}")

    if case["status"] in ("ready_for_review", "under_review"):
        render_decision_form(case_id)


# ============================================================
# 7. Main
# ============================================================

render_sidebar()

if not is_logged_in():
    render_login()
else:
    st.title("🛡️ Conversation Risk Decision Support")
    tab_submit, tab_queue = st.tabs(["Submit Case", "Case Queue"])

    with tab_submit:
        render_submit_case()
    with tab_queue:
        render_case_queue()