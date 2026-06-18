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

# If this file is frontend/app.py, project root is one level above frontend.
# If you put app.py directly in the root, this still works.
PROJECT_ROOT = (
    CURRENT_FILE.parent.parent
    if CURRENT_FILE.parent.name == "frontend"
    else CURRENT_FILE.parent
)

SRC_DIR = PROJECT_ROOT / "src"

if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# 1. Streamlit page config
# ============================================================

st.set_page_config(
    page_title="Hybrid Conversation Risk Detection",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# 2. Small utility functions
# ============================================================

def tensor_to_float(value: Any) -> float | None:
    """
    Converts torch tensor / numpy value / Python number into float.
    Returns None if the value is missing.
    """
    if value is None:
        return None

    try:
        # torch.Tensor
        if hasattr(value, "detach"):
            value = value.detach().cpu()

        if hasattr(value, "item"):
            return float(value.item())

        return float(value)
    except Exception:
        return None


def tensor_to_list(value: Any) -> list[float]:
    """
    Converts torch tensor / numpy array / list into Python list of floats.
    """
    if value is None:
        return []

    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()

        if hasattr(value, "tolist"):
            raw = value.tolist()
        else:
            raw = list(value)

        if isinstance(raw, (int, float)):
            return [float(raw)]

        return [float(x) for x in raw]
    except Exception:
        return []


def display_score(value: float | None) -> str:
    """
    Shows scores nicely in metric cards.
    """
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def risk_level(overall_risk: float | None) -> str:
    """
    Simple frontend display label.
    The final thresholds should be tuned later with validation data.
    """
    if overall_risk is None:
        return "Pending"

    if overall_risk >= 0.70:
        return "High"
    if overall_risk >= 0.40:
        return "Medium"
    return "Low"


def parse_messages(text: str, input_mode: str) -> list[dict[str, Any]]:
    """
    Converts raw text into a list of messages.

    Conversation mode:
        UserA: hello
        UserB: hi

    Single Text / Letter mode:
        The whole input becomes one message.
    """
    text = text.strip()

    if not text:
        return []

    if input_mode == "Single Text / Letter":
        return [
            {
                "speaker": "unknown",
                "text": text,
                "relative_time": 0.0,
            }
        ]

    messages = []

    for idx, line in enumerate(text.splitlines()):
        line = line.strip()

        if not line:
            continue

        # Supports:
        # UserA: message
        # UserA：message
        match = re.match(r"^([^:：]{1,40})[:：]\s*(.+)$", line)

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
    """
    Builds the project's ConversationWindow object.

    This function is already written to match your project folder:

        src/risk_detection/conversation.py

    It expects:
        ConversationWindow(k=...)
        Message(speaker_id=..., text=..., relative_time=...)
    """
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
# 3. Placeholder model loading
# ============================================================

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    TODO:
    Later, after your real model / calibrated model is ready,
    replace this function with your actual loading code.

    Example future structure:

        from risk_detection.model.integrated_pipeline import IntegratedInferencePipeline
        import torch

        pipeline = IntegratedInferencePipeline.from_pretrained()

        checkpoint_dir = PROJECT_ROOT / "checkpoints"

        pipeline.risk_fusion.load_state_dict(
            torch.load(checkpoint_dir / "risk_fusion.pt", map_location="cpu")
        )

        pipeline.emotion_score_head.load_state_dict(
            torch.load(checkpoint_dir / "emotion_score_head.pt", map_location="cpu")
        )

        return pipeline

    For now, return None so the frontend can run without fake model scores.
    """
    return None


# ============================================================
# 4. Result conversion
# ============================================================

def empty_dashboard_result(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Used before the real backend is connected.
    No fake scores. Everything missing is None or empty.
    """
    return {
        "analysis_available": False,
        "status": "Model pipeline is not connected yet. Scores are waiting for backend integration.",

        "overall_risk": None,
        "risk_level": "Pending",
        "safety_score": None,
        "emotion_score": None,

        "component_scores": {
            "cyberbullying": None,
            "grooming": None,
            "early_predator": None,
            "rule_score": None,
        },

        "emotion_report": {
            "fear": None,
            "sadness": None,
            "anger": None,
            "distress": None,
            "dependency": None,
        },

        "historical_state": {
            "accumulated_risk": None,
            "risk_trend": None,
            "risk_trend_label": "Pending",
            "persistence": None,
        },

        "confidence": None,
        "uncertainty": None,
        "human_review_required": None,

        "evidence_messages": {
            "cyberbullying": [],
            "conversation": [],
            "rule": [],
            "emotion": [],
        },

        "messages": messages,

        "limitations": [
            "This is a decision-support system. It does not automatically accuse, diagnose, report, or enforce.",
            "Evidence should be interpreted by a human analyst.",
            "Scores are unavailable until the backend model pipeline is connected.",
        ],
    }


def evidence_to_dict(evidence: Any) -> dict[str, list[int]]:
    """
    Converts your EvidenceBundle into frontend-friendly message numbers.

    Your backend currently returns zero-based indices.
    The frontend displays messages starting from 1, so we add +1.
    """
    if evidence is None:
        return {
            "cyberbullying": [],
            "conversation": [],
            "rule": [],
            "emotion": [],
        }

    output = {}

    for key in ["cyberbullying", "conversation", "rule", "emotion"]:
        raw_indices = getattr(evidence, key, []) or []

        converted = []
        for idx in raw_indices:
            try:
                converted.append(int(idx) + 1)
            except Exception:
                pass

        output[key] = converted

    return output


def result_to_dashboard_dict(raw_result: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Converts IntegratedInferenceResult into a frontend dictionary.

    This is where the Streamlit frontend connects to your project backend.

    Current backend fields:
        raw_result.safety_score
        raw_result.emotion_score
        raw_result.overall_score
        raw_result.component_scores
        raw_result.historical_state
        raw_result.risk_trend_label
        raw_result.evidence
        raw_result.uncertainty_estimate
        raw_result.human_review_required
        raw_result.limitations

    Future optional field:
        raw_result.mapped_emotions or raw_result.emotion_vector
    """
    overall_risk = tensor_to_float(getattr(raw_result, "overall_score", None))
    safety_score = tensor_to_float(getattr(raw_result, "safety_score", None))
    emotion_score = tensor_to_float(getattr(raw_result, "emotion_score", None))

    raw_components = getattr(raw_result, "component_scores", {}) or {}

    component_scores = {
        "cyberbullying": tensor_to_float(raw_components.get("cyberbullying")),
        "grooming": tensor_to_float(raw_components.get("grooming")),
        "early_predator": tensor_to_float(raw_components.get("early_predator")),
        "rule_score": tensor_to_float(raw_components.get("rule_score")),
    }

    historical_state = getattr(raw_result, "historical_state", None)

    historical_state_dict = {
        "accumulated_risk": tensor_to_float(getattr(historical_state, "accumulated_risk", None)),
        "risk_trend": tensor_to_float(getattr(historical_state, "risk_trend", None)),
        "risk_trend_label": getattr(raw_result, "risk_trend_label", "Pending"),
        "persistence": tensor_to_float(getattr(historical_state, "persistence", None)),
    }

    uncertainty_estimate = getattr(raw_result, "uncertainty_estimate", None)

    confidence = tensor_to_float(getattr(uncertainty_estimate, "confidence", None))
    uncertainty = tensor_to_float(getattr(uncertainty_estimate, "uncertainty", None))

    evidence_messages = evidence_to_dict(getattr(raw_result, "evidence", None))

    # ------------------------------------------------------------
    # Emotion report
    # ------------------------------------------------------------
    # Your current IntegratedInferenceResult does not return M_t yet.
    # Later, if you add one of these fields:
    #
    #     raw_result.mapped_emotions
    #     raw_result.emotion_vector
    #
    # this frontend will automatically show:
    # fear, sadness, anger, distress, dependency
    # ------------------------------------------------------------

    emotion_vector = None

    if hasattr(raw_result, "mapped_emotions"):
        emotion_vector = getattr(raw_result, "mapped_emotions")
    elif hasattr(raw_result, "emotion_vector"):
        emotion_vector = getattr(raw_result, "emotion_vector")

    emotion_values = tensor_to_list(emotion_vector)

    if len(emotion_values) >= 5:
        emotion_report = {
            "fear": emotion_values[0],
            "sadness": emotion_values[1],
            "anger": emotion_values[2],
            "distress": emotion_values[3],
            "dependency": emotion_values[4],
        }
    else:
        emotion_report = {
            "fear": None,
            "sadness": None,
            "anger": None,
            "distress": None,
            "dependency": None,
        }

    return {
        "analysis_available": True,
        "status": "Analysis completed.",

        "overall_risk": overall_risk,
        "risk_level": risk_level(overall_risk),
        "safety_score": safety_score,
        "emotion_score": emotion_score,

        "component_scores": component_scores,
        "emotion_report": emotion_report,
        "historical_state": historical_state_dict,

        "confidence": confidence,
        "uncertainty": uncertainty,
        "human_review_required": bool(getattr(raw_result, "human_review_required", False)),

        "evidence_messages": evidence_messages,
        "messages": messages,

        "limitations": list(getattr(raw_result, "limitations", [])),
    }


def analyze_text(
    text: str,
    input_mode: str,
    window_size: int,
) -> dict[str, Any]:
    """
    Main analysis function used by Streamlit.

    Right now:
        - Parses the input.
        - Builds the dashboard structure.
        - If pipeline is None, returns empty N/A result.

    Later:
        - load_pipeline() will return your real model.
        - pipeline.process(window) will produce real scores.
    """
    all_messages = parse_messages(text, input_mode)

    if not all_messages:
        return empty_dashboard_result([])

    if input_mode == "Conversation":
        model_messages = all_messages[-window_size:]
    else:
        model_messages = all_messages

    pipeline = load_pipeline()

    if pipeline is None:
        return empty_dashboard_result(model_messages)

    window = build_conversation_window(model_messages, window_size=window_size)
    raw_result = pipeline.process(window)

    return result_to_dashboard_dict(raw_result, model_messages)


# ============================================================
# 5. UI helpers
# ============================================================

def render_score_chart(title: str, score_dict: dict[str, float | None]) -> None:
    rows = []

    for name, value in score_dict.items():
        if value is not None:
            rows.append({"Name": name, "Score": value})

    st.subheader(title)

    if not rows:
        st.info("No score data available yet.")
        return

    df = pd.DataFrame(rows)
    st.bar_chart(df.set_index("Name"))


def render_messages_with_evidence(
    messages: list[dict[str, Any]],
    evidence_messages: dict[str, list[int]],
) -> None:
    st.subheader("Conversation / Text View")

    if not messages:
        st.info("No messages to display.")
        return

    highlighted_ids = set()

    for ids in evidence_messages.values():
        for idx in ids:
            highlighted_ids.add(idx)

    for idx, msg in enumerate(messages, start=1):
        speaker = msg.get("speaker", "unknown")
        text = msg.get("text", "")

        line = f"**Message {idx} — {speaker}:** {text}"

        if idx in highlighted_ids:
            st.warning(line)
        else:
            st.write(line)


def render_evidence(evidence_messages: dict[str, list[int]]) -> None:
    st.subheader("Evidence Messages")

    if not evidence_messages:
        st.info("No evidence available yet.")
        return

    st.json(evidence_messages)


def render_limitations(limitations: list[str]) -> None:
    st.subheader("Limitations")

    if not limitations:
        st.info("No limitations provided.")
        return

    for item in limitations:
        st.info(item)


# ============================================================
# 6. Sidebar sample inputs
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

You have ignored my warnings for too long. If you keep talking about me or try to report this, there will be consequences. I know where you usually go after school, and I can make sure everyone sees the messages I saved.

Do not show this letter to anyone. Do not tell your parents. If you do, things will get much worse for you.

This is your final warning."""


# ============================================================
# 7. Main Streamlit UI
# ============================================================

st.title("🛡️ Hybrid Conversation Risk Detection Dashboard")

st.caption(
    "Cyberbullying · Online Grooming · Early Predator Risk · Emotion Signals · Human Review Support"
)

st.sidebar.header("Settings")

input_mode = st.sidebar.radio(
    "Input Mode",
    ["Conversation", "Single Text / Letter"],
    horizontal=False,
)

window_size = st.sidebar.slider(
    "Rolling Window Size",
    min_value=1,
    max_value=30,
    value=12,
    step=1,
)

st.sidebar.divider()

if "conversation_input" not in st.session_state:
    st.session_state.conversation_input = ""

if st.sidebar.button("Load Sample Conversation"):
    st.session_state.conversation_input = SAMPLE_CONVERSATION

if st.sidebar.button("Load Sample Threat Letter"):
    st.session_state.conversation_input = SAMPLE_THREAT_LETTER

st.sidebar.divider()

st.sidebar.info(
    "Current version uses backend placeholders. "
    "Scores will show N/A until load_pipeline() is connected to your real model."
)

conversation_text = st.text_area(
    "Paste conversation or single text here",
    key="conversation_input",
    height=280,
    placeholder=(
        "Example:\n"
        "UserA: Hey, keep this between us...\n"
        "UserB: Why?\n"
        "UserA: Just trust me."
    ),
)

analyze_clicked = st.button("Analyze", type="primary")

if analyze_clicked:
    if not conversation_text.strip():
        st.warning("Please paste a conversation or text first.")
    else:
        try:
            result = analyze_text(
                text=conversation_text,
                input_mode=input_mode,
                window_size=window_size,
            )

            st.session_state["last_result"] = result

        except Exception as exc:
            st.error("Analysis failed.")
            st.exception(exc)


result = st.session_state.get("last_result")

if result is None:
    st.info("Paste text above and click Analyze.")
else:
    if result["analysis_available"]:
        st.success(result["status"])
    else:
        st.warning(result["status"])

    # ------------------------------------------------------------
    # Main score cards
    # ------------------------------------------------------------

    st.subheader("Summary Scores")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Overall Risk", display_score(result["overall_risk"]))
    col2.metric("Safety Score", display_score(result["safety_score"]))
    col3.metric("Emotion Score", display_score(result["emotion_score"]))
    col4.metric("Risk Level", result["risk_level"])

    # ------------------------------------------------------------
    # Human review
    # ------------------------------------------------------------

    review_required = result["human_review_required"]

    if review_required is True:
        st.error("Human Review Required")
    elif review_required is False:
        st.success("No immediate human review required")
    else:
        st.info("Human review decision is pending because model output is not available yet.")

    # ------------------------------------------------------------
    # Confidence and uncertainty
    # ------------------------------------------------------------

    st.subheader("Confidence / Uncertainty")

    col5, col6 = st.columns(2)

    col5.metric("Confidence", display_score(result["confidence"]))
    col6.metric("Uncertainty", display_score(result["uncertainty"]))

    # ------------------------------------------------------------
    # Component scores
    # ------------------------------------------------------------

    render_score_chart(
        "Component Scores",
        result["component_scores"],
    )

    # ------------------------------------------------------------
    # Emotion report
    # ------------------------------------------------------------

    render_score_chart(
        "Emotion Report",
        result["emotion_report"],
    )

    # ------------------------------------------------------------
    # Historical state
    # ------------------------------------------------------------

    st.subheader("Historical State")

    historical_state = result["historical_state"]

    col7, col8, col9, col10 = st.columns(4)

    col7.metric("Accumulated Risk", display_score(historical_state["accumulated_risk"]))
    col8.metric("Risk Trend", display_score(historical_state["risk_trend"]))
    col9.metric("Trend Label", historical_state["risk_trend_label"])
    col10.metric("Persistence", display_score(historical_state["persistence"]))

    # ------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------

    render_evidence(result["evidence_messages"])

    # ------------------------------------------------------------
    # Message view
    # ------------------------------------------------------------

    render_messages_with_evidence(
        messages=result["messages"],
        evidence_messages=result["evidence_messages"],
    )

    # ------------------------------------------------------------
    # Single text warning
    # ------------------------------------------------------------

    if input_mode == "Single Text / Letter":
        st.info(
            "Single-text mode is best for threat, rule, cyberbullying, and emotion-risk analysis. "
            "Grooming and early-detection scores may be less reliable because they depend on multi-turn context."
        )

    # ------------------------------------------------------------
    # Limitations
    # ------------------------------------------------------------

    render_limitations(result["limitations"])