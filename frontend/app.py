"""Streamlit frontend for the USAII Hybrid Conversation Risk Detection prototype.

This file intentionally changes only the frontend layer. It can run without the
real backend connected, and it can optionally call IntegratedInferencePipeline
once the model dependencies / checkpoints are available.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# ============================================================
# 0. Project path setup
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "frontend" else CURRENT_FILE.parent
SRC_DIR = PROJECT_ROOT / "src"

if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# 1. Page config
# ============================================================

st.set_page_config(
    page_title="Hybrid Conversation Risk Detection",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# 2. Constants that match report v15
# ============================================================

INPUT_MODE_CONVERSATION = "Conversation"
INPUT_MODE_SINGLE_TEXT = "Single Text / Letter"

MAPPED_EMOTION_NAMES = ["fear", "sadness", "anger", "distress", "dependency"]
COMPONENT_SCORE_NAMES = ["cyberbullying", "grooming", "rule_score"]
EVIDENCE_TYPES = ["cyberbullying", "conversation", "rule", "emotion"]

FRONTEND_ONLY_STATUS = "frontend_only_placeholder"
BACKEND_STATUS = "illustrative_unvalidated_example"


# ============================================================
# 3. Conversion helpers
# ============================================================


def tensor_to_float(value: Any) -> float | None:
    """Convert torch/numpy/Python scalar values into float for the UI."""
    if value is None:
        return None

    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()

        if hasattr(value, "item"):
            return float(value.item())

        return float(value)
    except Exception:
        return None


def tensor_to_flat_list(value: Any) -> list[float]:
    """Convert tensor/numpy/list values into a flat Python float list."""
    if value is None:
        return []

    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()

        if hasattr(value, "tolist"):
            value = value.tolist()
    except Exception:
        return []

    def _flatten(x: Any) -> list[Any]:
        if isinstance(x, (list, tuple)):
            out: list[Any] = []
            for item in x:
                out.extend(_flatten(item))
            return out
        return [x]

    values: list[float] = []

    for item in _flatten(value):
        try:
            values.append(float(item))
        except Exception:
            pass

    return values


def display_score(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def frontend_risk_level(overall_risk: float | None) -> str:
    """Readable bucket used only when backend dashboard output is unavailable."""
    if overall_risk is None:
        return "pending"

    if overall_risk >= 0.70:
        return "high"

    if overall_risk >= 0.40:
        return "medium"

    return "low"


def coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    try:
        if hasattr(value, "item"):
            return bool(value.item())

        return bool(value)
    except Exception:
        return None


def one_index_evidence(evidence: dict[str, Any] | None) -> dict[str, list[int]]:
    """Backend evidence indices are zero-based; the UI displays Message 1, 2, ..."""
    output: dict[str, list[int]] = {name: [] for name in EVIDENCE_TYPES}

    if not evidence:
        return output

    for name in EVIDENCE_TYPES:
        raw_items = evidence.get(name, []) if isinstance(evidence, dict) else []
        converted: list[int] = []

        for item in raw_items or []:
            try:
                converted.append(int(item) + 1)
            except Exception:
                pass

        output[name] = sorted(set(converted))

    return output


# ============================================================
# 4. Input parsing
# ============================================================


def parse_messages(text: str, input_mode: str) -> list[dict[str, Any]]:
    """Turn either a multi-turn conversation or one single text into messages."""
    text = text.strip()

    if not text:
        return []

    if input_mode == INPUT_MODE_SINGLE_TEXT:
        return [
            {
                "speaker": "single_text",
                "text": text,
                "relative_time": 0.0,
            }
        ]

    messages: list[dict[str, Any]] = []

    for idx, line in enumerate(text.splitlines()):
        line = line.strip()

        if not line:
            continue

        match = re.match(r"^([^:：]{1,60})[:：]\s*(.+)$", line)

        if match:
            speaker = match.group(1).strip()
            message_text = match.group(2).strip()
        else:
            speaker = "unknown"
            message_text = line

        messages.append(
            {
                "speaker": speaker,
                "text": message_text,
                "relative_time": float(idx),
            }
        )

    return messages


def build_conversation_window(messages: list[dict[str, Any]], window_size: int):
    """Build the project's ConversationWindow without changing backend code."""
    from risk_detection.conversation import ConversationWindow, Message

    window = ConversationWindow(k=window_size)

    for msg in messages:
        window.add(
            Message(
                speaker_id=str(msg["speaker"]),
                text=str(msg["text"]),
                relative_time=float(msg["relative_time"]),
            )
        )

    return window


# ============================================================
# 5. Optional backend loading
# ============================================================


@st.cache_resource(show_spinner=False)
def load_pipeline(enable_backend: bool):
    """Load backend only when the sidebar toggle is enabled.

    Report v15 says several model/fusion outputs are currently implemented
    but untrained or uncalibrated. For that reason the default UI mode is
    frontend-only: it displays the expected dashboard structure without
    pretending unvalidated scores are available.
    """
    if not enable_backend:
        return None

    from risk_detection.model import IntegratedInferencePipeline

    return IntegratedInferencePipeline.from_pretrained()


# ============================================================
# 6. Dashboard result normalization
# ============================================================


def empty_dashboard_result(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "analysis_available": False,
        "output_status": FRONTEND_ONLY_STATUS,
        "status_message": "Frontend is ready. Backend pipeline is not connected in this run, so scores are N/A.",
        "overall_risk": None,
        "risk_level": "pending",
        "summary_scores": {
            "safety_score": None,
            "emotion_score": None,
            "overall_score": None,
        },
        "component_scores": {name: None for name in COMPONENT_SCORE_NAMES},
        "early_warning": {
            "triggered": None,
            "method": "persistence_based_baseline",
            "accumulated_risk": None,
            "risk_trend": None,
            "risk_trend_label": "pending",
            "persistence": None,
        },
        "emotion_report": {
            "emotion_score": None,
            "score_status": "unavailable_until_backend_connected",
            "signals": {name: None for name in MAPPED_EMOTION_NAMES},
            "primary_emotions": [],
            "interpretation": "No emotion analysis is available until the backend is connected.",
            "model_type": "GoEmotions mapping + prototype emotion layer",
        },
        "evidence_messages": {name: [] for name in EVIDENCE_TYPES},
        "uncertainty": None,
        "confidence": None,
        "human_review_required": None,
        "messages": messages,
        "limitations": [
            "Frontend-only mode: no backend model scores have been computed.",
            "This is a decision-support system. It does not automatically accuse, diagnose, report, or enforce.",
            "Evidence and review recommendations must be interpreted by a human analyst.",
        ],
    }


def emotion_report_from_raw_result(raw_result: Any) -> dict[str, Any]:
    """Fallback when risk_detection.model.to_dashboard_dict is unavailable."""
    vector = getattr(raw_result, "mapped_emotions", None)
    values = tensor_to_flat_list(vector)

    signals = {name: None for name in MAPPED_EMOTION_NAMES}

    if len(values) >= len(MAPPED_EMOTION_NAMES):
        signals = dict(zip(MAPPED_EMOTION_NAMES, values[: len(MAPPED_EMOTION_NAMES)]))

    numeric_signals = {key: value for key, value in signals.items() if value is not None}
    primary = sorted(numeric_signals, key=numeric_signals.get, reverse=True)[:3]

    if primary:
        interpretation = f"Elevated emotional signals to inspect: {', '.join(primary)}."
    else:
        interpretation = "No mapped emotion signals are available."

    return {
        "emotion_score": tensor_to_float(getattr(raw_result, "emotion_score", None)),
        "score_status": "illustrative_placeholder",
        "signals": signals,
        "primary_emotions": primary,
        "interpretation": interpretation,
        "model_type": "uncalibrated_prototype_emotion_layer",
    }


def normalize_dashboard_dict(
    dashboard: dict[str, Any],
    raw_result: Any | None,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Normalize backend Section-16 JSON into the exact shape the frontend uses."""
    summary = dashboard.get("summary_scores", {}) or {}

    if dashboard.get("emotion_report") is not None:
        emotion_report = dashboard.get("emotion_report") or {}
    elif raw_result is not None:
        emotion_report = emotion_report_from_raw_result(raw_result)
    else:
        emotion_report = empty_dashboard_result(messages)["emotion_report"]

    component_scores = dashboard.get("component_scores", {}) or {}
    component_scores = {
        name: tensor_to_float(component_scores.get(name))
        for name in COMPONENT_SCORE_NAMES
    }

    early_warning = dashboard.get("early_warning", {}) or {}

    overall_score = tensor_to_float(summary.get("overall_score", dashboard.get("overall_risk")))

    normalized = {
        "analysis_available": True,
        "output_status": dashboard.get("output_status", BACKEND_STATUS),
        "status_message": (
            "Backend analysis completed. Treat uncalibrated prototype outputs as illustrative "
            "unless the status says validated."
        ),
        "overall_risk": tensor_to_float(dashboard.get("overall_risk", overall_score)),
        "risk_level": dashboard.get("risk_level") or frontend_risk_level(overall_score),
        "summary_scores": {
            "safety_score": tensor_to_float(summary.get("safety_score")),
            "emotion_score": tensor_to_float(summary.get("emotion_score")),
            "overall_score": overall_score,
        },
        "component_scores": component_scores,
        "early_warning": {
            "triggered": coerce_bool(early_warning.get("triggered")),
            "method": early_warning.get("method", "persistence_based_baseline"),
            "accumulated_risk": tensor_to_float(early_warning.get("accumulated_risk")),
            "risk_trend": tensor_to_float(early_warning.get("risk_trend")),
            "risk_trend_label": early_warning.get("risk_trend_label", "pending"),
            "persistence": tensor_to_float(early_warning.get("persistence")),
        },
        "emotion_report": {
            "emotion_score": tensor_to_float(emotion_report.get("emotion_score")),
            "score_status": emotion_report.get("score_status", "illustrative_placeholder"),
            "signals": {
                name: tensor_to_float((emotion_report.get("signals") or {}).get(name))
                for name in MAPPED_EMOTION_NAMES
            },
            "primary_emotions": list(emotion_report.get("primary_emotions") or []),
            "interpretation": emotion_report.get("interpretation", ""),
            "model_type": emotion_report.get("model_type", "uncalibrated_prototype_emotion_layer"),
        },
        "evidence_messages": one_index_evidence(dashboard.get("evidence_messages")),
        "uncertainty": tensor_to_float(dashboard.get("uncertainty")),
        "confidence": tensor_to_float(dashboard.get("confidence")),
        "human_review_required": coerce_bool(dashboard.get("human_review_required")),
        "messages": messages,
        "limitations": list(getattr(raw_result, "limitations", [])) if raw_result is not None else [],
    }

    if not normalized["limitations"]:
        normalized["limitations"] = [
            "This is a decision-support system, not an automatic enforcement system.",
            "Some current prototype scores may be untrained or uncalibrated placeholders.",
            "All evidence must be interpreted by a human analyst.",
        ]

    return normalized


def raw_result_to_dashboard(raw_result: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Use backend's v15 Section-16 serializer when available."""
    try:
        from risk_detection.model import to_dashboard_dict

        dashboard = to_dashboard_dict(raw_result)
        return normalize_dashboard_dict(dashboard, raw_result, messages)

    except Exception:
        dashboard = {
            "output_status": BACKEND_STATUS,
            "overall_risk": tensor_to_float(getattr(raw_result, "overall_score", None)),
            "risk_level": frontend_risk_level(tensor_to_float(getattr(raw_result, "overall_score", None))),
            "summary_scores": {
                "safety_score": tensor_to_float(getattr(raw_result, "safety_score", None)),
                "emotion_score": tensor_to_float(getattr(raw_result, "emotion_score", None)),
                "overall_score": tensor_to_float(getattr(raw_result, "overall_score", None)),
            },
            "component_scores": getattr(raw_result, "component_scores", {}) or {},
            "early_warning": (
                getattr(raw_result, "early_warning", None).__dict__
                if getattr(raw_result, "early_warning", None) is not None
                else {}
            ),
            "emotion_report": emotion_report_from_raw_result(raw_result),
            "evidence_messages": {
                name: getattr(getattr(raw_result, "evidence", None), name, [])
                for name in EVIDENCE_TYPES
            },
            "uncertainty": tensor_to_float(
                getattr(getattr(raw_result, "uncertainty_estimate", None), "uncertainty", None)
            ),
            "confidence": tensor_to_float(
                getattr(getattr(raw_result, "uncertainty_estimate", None), "confidence", None)
            ),
            "human_review_required": getattr(raw_result, "human_review_required", None),
        }

        return normalize_dashboard_dict(dashboard, raw_result, messages)


def analyze_text(
    text: str,
    input_mode: str,
    window_size: int,
    enable_backend: bool,
) -> dict[str, Any]:
    all_messages = parse_messages(text, input_mode)

    if not all_messages:
        return empty_dashboard_result([])

    if input_mode == INPUT_MODE_CONVERSATION:
        model_messages = all_messages[-window_size:]
    else:
        model_messages = all_messages

    pipeline = load_pipeline(enable_backend)

    if pipeline is None:
        return empty_dashboard_result(model_messages)

    window = build_conversation_window(model_messages, window_size=window_size)
    raw_result = pipeline.process(window)

    return raw_result_to_dashboard(raw_result, model_messages)


# ============================================================
# 7. Rendering helpers
# ============================================================


def render_status(result: dict[str, Any]) -> None:
    status = result.get("output_status", "unknown")

    if result.get("analysis_available"):
        st.success(result.get("status_message", "Analysis completed."))
    else:
        st.warning(result.get("status_message", "Backend is not connected."))

    if status == "illustrative_unvalidated_example":
        st.info(
            "Output status: illustrative / unvalidated prototype. Report v15 says Grooming, "
            "EmotionScoreHead, RiskFusion, and calibration weights are not yet validated, "
            "so these scores should not be treated as operational conclusions."
        )
    elif status == FRONTEND_ONLY_STATUS:
        st.info(
            "Output status: frontend-only placeholder. "
            "The page layout is ready, but no model output has been computed."
        )
    else:
        st.info(f"Output status: {status}")


def render_score_cards(result: dict[str, Any]) -> None:
    summary = result["summary_scores"]

    st.subheader("Summary Scores")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Overall Risk", display_score(summary.get("overall_score") or result.get("overall_risk")))
    col2.metric("Safety Score", display_score(summary.get("safety_score")))
    col3.metric("Emotion Score", display_score(summary.get("emotion_score")))
    col4.metric("Risk Level", str(result.get("risk_level", "pending")).upper())


def render_human_review(result: dict[str, Any]) -> None:
    st.subheader("Human Review Recommendation")

    review = result.get("human_review_required")

    if review is True:
        st.error("Human Review Required")
    elif review is False:
        st.success("No immediate human review required")
    else:
        st.info("Pending — no backend review decision is available yet.")

    st.caption(
        "In the current prototype, the rule-based review subset is the operable part today: "
        "rule_score ≥ 0.8 or a direct threat-phrase rule trigger. Model-score-based routing "
        "remains unvalidated until calibration is finished."
    )


def render_score_chart(title: str, score_dict: dict[str, float | None]) -> None:
    st.subheader(title)

    rows = [
        {"Name": key, "Score": value}
        for key, value in score_dict.items()
        if value is not None
    ]

    if not rows:
        st.info("No score data available yet.")
        return

    df = pd.DataFrame(rows).set_index("Name")
    st.bar_chart(df)


def render_early_warning(early_warning: dict[str, Any]) -> None:
    st.subheader("Persistence-Based Early Warning")

    triggered = early_warning.get("triggered")

    if triggered is True:
        st.error("Early warning triggered")
    elif triggered is False:
        st.success("Early warning not triggered")
    else:
        st.info("Early warning pending — backend not connected.")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accumulated Risk", display_score(early_warning.get("accumulated_risk")))
    col2.metric("Risk Trend", display_score(early_warning.get("risk_trend")))
    col3.metric("Trend Label", str(early_warning.get("risk_trend_label", "pending")))
    col4.metric("Persistence", display_score(early_warning.get("persistence")))

    st.caption(f"Method: {early_warning.get('method', 'persistence_based_baseline')}")


def render_emotion_report(emotion_report: dict[str, Any]) -> None:
    st.subheader("Emotion Report")

    col1, col2 = st.columns(2)

    col1.metric("Emotion Score", display_score(emotion_report.get("emotion_score")))
    col2.metric("Score Status", str(emotion_report.get("score_status", "pending")))

    st.caption(f"Model type: {emotion_report.get('model_type', 'unknown')}")

    signals = emotion_report.get("signals", {}) or {}

    render_score_chart("Mapped Emotion Signals", signals)

    primary = emotion_report.get("primary_emotions") or []

    if primary:
        st.write("Primary emotions:", ", ".join(primary))
    else:
        st.write("Primary emotions: N/A")

    interpretation = emotion_report.get("interpretation")

    if interpretation:
        st.info(interpretation)


def render_confidence(result: dict[str, Any]) -> None:
    st.subheader("Confidence / Uncertainty")

    col1, col2 = st.columns(2)

    col1.metric(
        "Confidence",
        display_score(result.get("confidence")),
        help="Prediction stability, not correctness probability.",
    )
    col2.metric("Uncertainty", display_score(result.get("uncertainty")))


def render_evidence(evidence_messages: dict[str, list[int]]) -> None:
    st.subheader("Evidence Messages")

    st.caption(
        "Message numbers below are 1-based for readability. "
        "Attention evidence indicates model focus, not proof."
    )

    cols = st.columns(4)

    for col, evidence_type in zip(cols, EVIDENCE_TYPES):
        ids = evidence_messages.get(evidence_type, []) or []

        with col:
            st.metric(evidence_type.replace("_", " ").title(), len(ids))
            st.write(ids if ids else "—")

    with st.expander("Raw evidence JSON"):
        st.json(evidence_messages)


def render_messages(
    messages: list[dict[str, Any]],
    evidence_messages: dict[str, list[int]],
) -> None:
    st.subheader("Conversation / Text View")

    if not messages:
        st.info("No messages to display.")
        return

    highlighted: set[int] = set()
    evidence_by_id: dict[int, list[str]] = {}

    for evidence_type, ids in evidence_messages.items():
        for msg_id in ids:
            highlighted.add(msg_id)
            evidence_by_id.setdefault(msg_id, []).append(evidence_type)

    for idx, msg in enumerate(messages, start=1):
        speaker = msg.get("speaker", "unknown")
        text = msg.get("text", "")
        evidence_tags = evidence_by_id.get(idx, [])
        label = f"Message {idx} — {speaker}"

        if idx in highlighted:
            st.warning(f"**{label}**  · evidence: {', '.join(evidence_tags)}\n\n{text}")
        else:
            st.write(f"**{label}**\n\n{text}")


def render_limitations(limitations: list[str]) -> None:
    st.subheader("Limitations")

    for item in limitations:
        st.info(item)


# ============================================================
# 8. Sample inputs
# ============================================================

SAMPLE_CONVERSATION = """UserA: Hey, are you online right now?
UserB: Yeah, I just finished school.
UserA: Good. Don’t tell your parents we talk this much, they probably won’t understand.
UserB: Why not? I don’t think it’s a big deal.
UserA: Because people always judge things. You can trust me more than your friends anyway.
UserC: Nobody even likes you in class. Everyone thinks you are weird.
UserB: Please stop saying that.
UserC: If you tell the teacher, I’ll show everyone the screenshots.
UserA: Ignore them. I’m the only one who actually cares about you.
UserB: I feel like nobody else listens to me.
UserA: Exactly. That’s why you should talk to me privately. Add me on another app so we can chat without anyone seeing.
UserB: I’m not sure.
UserA: Come on, don’t be difficult. Just keep this between us."""

SAMPLE_THREAT_LETTER = """To whoever reads this,

You have ignored my warnings for too long. If you keep talking about me or try to report this, there will be consequences. I saved screenshots and I can make things much worse for you.

Do not show this letter to anyone. Do not tell your parents. If you do, things will get much worse for you.

This is your final warning."""

SAMPLE_SAFE_CONVERSATION = """StudentA: Did you finish the project slides?
StudentB: Almost. I added the model architecture diagram.
StudentA: Great. I’ll review the dashboard section tonight.
StudentB: Thanks. We should also mention the limitations clearly.
StudentA: Agreed. Let’s keep the demo simple and explain that it is decision-support only."""


# ============================================================
# 9. Main UI
# ============================================================

st.title("🛡️ Hybrid Conversation Risk Detection Dashboard")

st.caption(
    "Cyberbullying · Online Grooming · Persistence-Based Early Warning · Emotion Signals · Human Review Support"
)

with st.sidebar:
    st.header("Settings")

    input_mode = st.radio(
        "Input Mode",
        [INPUT_MODE_CONVERSATION, INPUT_MODE_SINGLE_TEXT],
        horizontal=False,
    )

    window_size = st.slider(
        "Rolling Window Size",
        min_value=1,
        max_value=30,
        value=12,
        step=1,
    )

    enable_backend = st.toggle(
        "Try real backend pipeline",
        value=False,
        help=(
            "Off by default so the frontend can run without downloading Hugging Face models. "
            "Turn on only when backend dependencies/checkpoints are ready."
        ),
    )

    st.divider()

    st.caption("Sample Inputs")

    if "conversation_input" not in st.session_state:
        st.session_state.conversation_input = ""

    if st.button("Load Sample Conversation"):
        st.session_state.conversation_input = SAMPLE_CONVERSATION
        st.session_state.last_result = None

    if st.button("Load Sample Threat Letter"):
        st.session_state.conversation_input = SAMPLE_THREAT_LETTER
        st.session_state.last_result = None

    if st.button("Load Safe Conversation"):
        st.session_state.conversation_input = SAMPLE_SAFE_CONVERSATION
        st.session_state.last_result = None

    st.divider()

    st.info(
        "Report v15 status: the dashboard is implemented, but several model/fusion scores "
        "are currently illustrative until training and calibration are complete."
    )

conversation_text = st.text_area(
    "Paste conversation or single text here",
    key="conversation_input",
    height=300,
    placeholder="UserA: Hey, keep this between us...\nUserB: Why?\nUserA: Just trust me.",
)

analyze_clicked = st.button("Analyze", type="primary")

if analyze_clicked:
    if not conversation_text.strip():
        st.warning("Please paste a conversation or text first.")
    else:
        try:
            st.session_state.last_result = analyze_text(
                text=conversation_text,
                input_mode=input_mode,
                window_size=window_size,
                enable_backend=enable_backend,
            )
        except Exception as exc:
            st.error("Analysis failed.")
            st.exception(exc)

result = st.session_state.get("last_result")

if result is None:
    st.info("Paste text above and click Analyze.")
else:
    render_status(result)
    render_score_cards(result)
    render_human_review(result)
    render_confidence(result)

    tab_scores, tab_emotion, tab_evidence, tab_text, tab_limits = st.tabs(
        ["Scores", "Emotion", "Evidence", "Messages", "Limitations"]
    )

    with tab_scores:
        render_score_chart("Component Scores", result["component_scores"])
        render_early_warning(result["early_warning"])

    with tab_emotion:
        render_emotion_report(result["emotion_report"])

    with tab_evidence:
        render_evidence(result["evidence_messages"])

    with tab_text:
        render_messages(result["messages"], result["evidence_messages"])

        if input_mode == INPUT_MODE_SINGLE_TEXT:
            st.info(
                "Single-text mode is most useful for threat/rule/cyberbullying/emotion inspection. "
                "Grooming and early-warning logic are more reliable with multi-turn context."
            )

    with tab_limits:
        render_limitations(result["limitations"])