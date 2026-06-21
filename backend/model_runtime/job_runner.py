"""Builds a fresh per-job pipeline around the shared heavy ModelComponents
and runs one case's analysis (v6 Section 3: AI Analysis Workers).

Multi-window semantics: a case's messages are split into successive,
non-overlapping ConversationWindows of at most `window_size` messages each
(build_windows below), processed in order through ONE pipeline instance.
HistoricalStateUpdater/EarlyWarningTracker are stateful by design (see their
own docstrings) and persist across successive `process()` calls *within
this one case's analysis*, so each window becomes one time step in the same
Et/Tt/Pt recurrence and the same latched early warning -- but a fresh
pipeline (and fresh updater/tracker) is built per call to run_analysis, so
risk history never leaks between cases (per the plan's correctness note).

Windows are deliberately non-overlapping (stride == window_size): an
overlapping window would feed the same messages' r_t into the Section 8
accumulators more than once, inflating accumulated risk and persistence in
a way that recurrence was never designed to handle -- it assumes each time
step is genuinely new information, not a re-observed shifted view of the
same messages.

This whole multi-window aggregation (case-level risk = highest-risk
window, human_review_required = OR across windows, evidence merged across
windows) is a proposed extension with no labeled-outcome validation:
Variant A/B/C's measured precision/recall (Section 6/19.1) are all
single-window, whole-conversation numbers. See
_multi_window_coverage_limitations' disclosure string, surfaced verbatim
to analysts via data_limitations whenever a case actually needed more than
one window.
"""

from dataclasses import dataclass

from risk_detection import ConversationWindow, Message
from risk_detection.model import EarlyWarningTracker, HistoricalStateUpdater, IntegratedInferencePipeline, IntegratedInferenceResult
from risk_detection.signals.llm_safety import LLMSafetySignals
from risk_detection.signals.rules import RULE_SIGNAL_NAMES, RuleEvidence, RuleSignalExtractor, RuleSignals
from risk_detection.signals.safety_features import SafetyFeatures

from backend.config import settings
from backend.model_runtime.loader import ModelComponents, get_model_components

RISK_LEVEL_THRESHOLDS = (
    # Illustrative thresholds (the doc specifies the 4 risk_level values but
    # not where to draw the lines against a continuous score) -- same
    # "illustrative default" framing the doc uses elsewhere (Section 6.1, 10).
    (0.75, "critical"),
    (0.50, "high"),
    (0.25, "medium"),
)


@dataclass(frozen=True)
class JobMessage:
    speaker_local_id: str
    redacted_content: str | None
    message_sequence: int


@dataclass(frozen=True)
class AnalysisOutcome:
    result: IntegratedInferenceResult  # the highest-overall_score window's result
    rule_evidence: RuleEvidence  # merged across ALL windows -- indices are message_sequence numbers, NOT window-local positions
    safety_features: SafetyFeatures  # merged across ALL windows (max per LLM signal, OR per rule signal)
    window_message_sequences: tuple[int, ...]  # the representative window's own position -> message_sequence mapping, for translating result.evidence
    analyzed_message_sequences: tuple[int, ...]  # union of every message_sequence covered by any window, sorted
    extra_limitations: tuple[str, ...]
    window_count: int


@dataclass(frozen=True)
class _WindowOutcome:
    result: IntegratedInferenceResult
    rule_evidence: RuleEvidence  # window-local positions
    safety_features: SafetyFeatures
    window_message_sequences: tuple[int, ...]
    token_overflow: bool


def risk_level_from_score(score: float) -> str:
    for threshold, label in RISK_LEVEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "low"


def build_windows(
    messages: list[JobMessage], window_size: int, stride: int | None = None
) -> list[tuple[ConversationWindow, tuple[int, ...]]]:
    """Splits a case's messages into successive ConversationWindows of at
    most window_size messages each, in submission order. Defaults to
    non-overlapping windows (stride=window_size) -- see this module's
    docstring for why overlap and the Section 8 state recurrence don't mix.
    """
    if not messages:
        return []

    stride = stride or window_size
    n = len(messages)
    windows: list[tuple[ConversationWindow, tuple[int, ...]]] = []
    start = 0
    while start < n:
        chunk = messages[start : start + window_size]
        window = ConversationWindow(k=window_size)
        for msg in chunk:
            window.add(Message(speaker_id=msg.speaker_local_id, text=msg.redacted_content or "", relative_time=0.0))
        windows.append((window, tuple(msg.message_sequence for msg in chunk)))
        if start + window_size >= n:
            break
        start += stride
    return windows


def _message_exceeds_encoder_budget(window: ConversationWindow, message_encoder) -> bool:
    tokenizer = message_encoder.tokenizer
    max_length = message_encoder.max_length
    return any(len(tokenizer(f"{m.speaker_id}: {m.text}")["input_ids"]) > max_length for m in window)


def _merge_safety_features(outcomes: list[_WindowOutcome]) -> SafetyFeatures:
    """LLM signals: max per dimension across windows (display-only -- Lt
    feeds no production score, Section 19.5 -- so the strongest signal seen
    anywhere in the case is the most informative single number to show).
    Rule signals: OR per rule across windows (a rule firing anywhere in the
    case should count, matching human_review_required's OR below)."""
    llm = [o.safety_features.llm_signals for o in outcomes]
    rules = [o.safety_features.rule_signals for o in outcomes]
    return SafetyFeatures(
        llm_signals=LLMSafetySignals(
            secrecy=max(s.secrecy for s in llm),
            isolation=max(s.isolation for s in llm),
            dependency=max(s.dependency for s in llm),
            sexual_escalation=max(s.sexual_escalation for s in llm),
            threat=max(s.threat for s in llm),
            coercion=max(s.coercion for s in llm),
        ),
        rule_signals=RuleSignals(
            secret_request=any(r.secret_request for r in rules),
            contact_migration=any(r.contact_migration for r in rules),
            age_reference=any(r.age_reference for r in rules),
            image_request=any(r.image_request for r in rules),
            threat_phrase=any(r.threat_phrase for r in rules),
        ),
    )


def _merge_rule_evidence_across_windows(outcomes: list[_WindowOutcome]) -> RuleEvidence:
    """Each window's RuleEvidence holds window-local positions (0..len(window)-1).
    Window 1's position 5 and window 2's position 5 refer to different
    messages, so positions must be translated to message_sequence via each
    window's OWN mapping before merging -- not merged as raw positions."""
    merged: dict[str, set[int]] = {name: set() for name in RULE_SIGNAL_NAMES}
    for outcome in outcomes:
        for rule_name, positions in outcome.rule_evidence.triggered_message_indices.items():
            for position in positions:
                if 0 <= position < len(outcome.window_message_sequences):
                    merged[rule_name].add(outcome.window_message_sequences[position])
    return RuleEvidence(triggered_message_indices={name: sorted(seqs) for name, seqs in merged.items()})


def _select_representative_window(outcomes: list[_WindowOutcome]) -> _WindowOutcome:
    """The window whose own overall_score is highest -- the case's headline
    risk_level/scores should reflect the worst moment observed, not
    whichever window happens to be last: Et's EMA recurrence (Section 8.2)
    decays past spikes, so the last window alone could understate a risk
    that already occurred (the latched early-warning catches this for
    Warningt, but overall_score/risk_level have no such latch)."""
    return max(outcomes, key=lambda o: o.result.overall_score.item())


def _multi_window_coverage_limitations(
    full_message_count: int, window_size: int, window_count: int, any_token_overflow: bool, encoder_max_length: int
) -> tuple[str, ...]:
    limitations = []

    if window_count > 1:
        limitations.append(
            f"This case's {full_message_count} messages were split into {window_count} sequential "
            f"{window_size}-message windows for analysis (RISK_PLATFORM_WINDOW_SIZE); case-level "
            "risk_level/scores reflect the single highest-risk window, human_review_required is true "
            "if ANY window triggered it, and Historical State (accumulated risk/trend/persistence) "
            "accumulates sequentially across windows in submission order. This multi-window "
            "aggregation is a proposed extension, not yet validated against labeled outcomes -- "
            "Variant A/B/C precision/recall (Section 6/19.1) were all measured on single-window, "
            "whole-conversation analysis."
        )

    if any_token_overflow:
        limitations.append(
            f"One or more messages in this case exceeded the trained text encoder's "
            f"{encoder_max_length}-token limit; the encoder and emotion classifier only analyzed the "
            f"first {encoder_max_length} tokens of each such message. The LLM safety-signal extractor "
            "and rule engine still scanned each message's full text."
        )

    return tuple(limitations)


def run_analysis(
    messages: list[JobMessage],
    components: ModelComponents | None = None,
    window_size: int | None = None,
) -> AnalysisOutcome:
    components = components or get_model_components()
    window_size = window_size if window_size is not None else settings.default_window_size

    pipeline = IntegratedInferencePipeline(
        message_encoder=components.message_encoder,
        conversation_encoder=components.conversation_encoder,
        cyberbullying_head=components.cyberbullying_head,
        grooming_head=components.grooming_head,
        emotion_classifier=components.emotion_classifier,
        emotion_score_head=components.emotion_score_head,
        risk_fusion=components.risk_fusion,
        historical_state_updater=HistoricalStateUpdater(),
        early_warning_tracker=EarlyWarningTracker(),
        safety_feature_extractor=components.safety_feature_extractor,
        dependency_extractor=components.dependency_extractor,
        grooming_message_encoder=components.grooming_message_encoder,
        grooming_conversation_encoder=components.grooming_conversation_encoder,
    )

    windows = build_windows(messages, window_size)
    rule_extractor: RuleSignalExtractor = components.safety_feature_extractor.rule_extractor

    window_outcomes: list[_WindowOutcome] = []
    for window, window_message_sequences in windows:
        # Sequential process() calls on the SAME pipeline share its
        # historical_state_updater/early_warning_tracker (stateful by
        # design), so this window becomes the next time step in this
        # case's accumulated risk/trend/persistence and latched warning.
        result = pipeline.process(window)
        window_outcomes.append(
            _WindowOutcome(
                result=result,
                rule_evidence=rule_extractor.extract_evidence(window),
                safety_features=components.safety_feature_extractor.extract(window),
                window_message_sequences=window_message_sequences,
                token_overflow=_message_exceeds_encoder_budget(window, components.message_encoder),
            )
        )

    representative = _select_representative_window(window_outcomes)

    # human_review_required isn't latched like Warningt -- a rule trigger
    # in an earlier, non-representative window must still flag the case,
    # so OR it in here rather than trusting the representative window's
    # own flag alone.
    representative.result.human_review_required = representative.result.human_review_required or any(
        o.result.human_review_required for o in window_outcomes
    )

    coverage_limitations = _multi_window_coverage_limitations(
        full_message_count=len(messages),
        window_size=window_size,
        window_count=len(windows),
        any_token_overflow=any(o.token_overflow for o in window_outcomes),
        encoder_max_length=components.message_encoder.max_length,
    )

    analyzed_sequences = sorted({seq for o in window_outcomes for seq in o.window_message_sequences})

    return AnalysisOutcome(
        result=representative.result,
        rule_evidence=_merge_rule_evidence_across_windows(window_outcomes),
        safety_features=_merge_safety_features(window_outcomes),
        window_message_sequences=representative.window_message_sequences,
        analyzed_message_sequences=tuple(analyzed_sequences),
        extra_limitations=components.extra_limitations + coverage_limitations,
        window_count=len(windows),
    )
