from dataclasses import dataclass

import torch

from ...conversation import ConversationWindow
from ...signals.emotional_dependency import EmotionalDependencyExtractor
from ...signals.rule_score import rule_safety_score
from ...signals.safety_features import SafetyFeatureExtractor
from ..encoder.aggregation import max_mean_top3
from ..encoder.conversation_encoder import ConversationEncoder
from ..heads.cyberbullying_head import CyberbullyingHead
from ..emotion.emotion_classifier import DEFAULT_GOEMOTIONS_MODEL, GoEmotionsClassifier
from ..emotion.emotion_mapping import DEFAULT_LAMBDA, map_emotions
from ..emotion.emotion_score_head import EmotionScoreHead
from .evidence import EvidenceBundle, extract_evidence
from ..heads.grooming_head import GroomingHead
from ..state.early_warning import DEFAULT_THRESHOLDS, EarlyWarningThresholds, EarlyWarningTracker
from ..state.historical_state import (
    DEFAULT_PERSISTENCE_WINDOW,
    HistoricalRiskState,
    HistoricalStateUpdater,
    trend_label,
)
from ..encoder.message_encoder import DEFAULT_ENCODER_NAME, MessageEncoder
from ..fusion.risk_fusion import PrototypeRiskFusion
from .uncertainty import (
    DEFAULT_MC_DROPOUT_PASSES,
    UncertaintyEstimate,
    currently_operable_review,
    enable_mc_dropout,
    mc_dropout_stats,
)

EARLY_WARNING_METHOD = "persistence_based_baseline"

LIMITATIONS = (
    "This is a decision-support system. It does not automatically label a "
    "person as an offender or victim, make a diagnosis, report a case, or "
    "trigger enforcement action.",
    "Attention-based evidence indicates model focus, not causal or legal proof.",
    "Confidence represents prediction stability, not correctness probability.",
    "The MC Dropout uncertainty estimate primarily captures uncertainty from "
    "the trainable safety and fusion components; it does not fully represent "
    "uncertainty in the frozen GoEmotions classifier or the emotion mapping layer.",
    "Safety Score, Overall Score, and the MC Dropout uncertainty estimate are "
    "illustrative: the Grooming Head, EmotionScoreHead, and RiskFusion have "
    "not yet been trained or calibrated (Section 19.5). Only the rule-based "
    "human review condition and the persistence-based early warning below "
    "are currently operable.",
    "All evidence must be interpreted by a human analyst.",
)


@dataclass
class EarlyWarningInfo:
    """Section 9.1/16: the dashboard's early_warning object."""

    triggered: bool
    method: str
    accumulated_risk: float
    risk_trend: float
    risk_trend_label: str
    persistence: float


@dataclass
class IntegratedInferenceResult:
    """Section 15 step 12: everything the dashboard needs for one
    Conversation Window / time step."""

    safety_score: torch.Tensor  # S_safety(t), prototype (no S_e)
    emotion_score: torch.Tensor  # S_emotion(t)
    mapped_emotions: torch.Tensor  # M_t = [Fear, Sadness, Anger, Distress, Dependency], Section 10.2
    overall_score: torch.Tensor  # R_t, deterministic forward pass
    component_scores: dict[str, torch.Tensor]  # cyberbullying, grooming, rule_score
    historical_state: HistoricalRiskState  # H_t, after this step's update
    risk_trend_label: str
    early_warning: EarlyWarningInfo
    evidence: EvidenceBundle
    uncertainty_estimate: UncertaintyEstimate  # R_hat_t, variance, uncertainty, confidence
    human_review_required: bool
    limitations: tuple[str, ...]


class IntegratedInferencePipeline:
    """Section 15: Current Prototype Inference Flow. Wires together every
    currently-trainable component into one pass over a rolling
    Conversation Window, computing each expensive step exactly once -- the
    LLM safety signal call, the shared message/conversation encoder, the
    GoEmotions classifier -- rather than letting the per-task pipelines
    from earlier sections each repeat them independently.

    The Early Detection Head (S_e, Section 9.3) is deliberately not part
    of this pipeline: it is untrained, and the report's current inference
    flow no longer computes S_e as a step (Section 15). Use
    `EarlyDetectionPipeline` directly once it has trained weights.

    `historical_state_updater` and `early_warning_tracker` are both
    stateful and persist across successive `process()` calls on the same
    conversation.
    """

    def __init__(
        self,
        message_encoder: MessageEncoder,
        conversation_encoder: ConversationEncoder,
        cyberbullying_head: CyberbullyingHead,
        grooming_head: GroomingHead,
        emotion_classifier: GoEmotionsClassifier,
        emotion_score_head: EmotionScoreHead,
        risk_fusion: PrototypeRiskFusion,
        historical_state_updater: HistoricalStateUpdater,
        early_warning_tracker: EarlyWarningTracker | None = None,
        safety_feature_extractor: SafetyFeatureExtractor | None = None,
        dependency_extractor: EmotionalDependencyExtractor | None = None,
        lam: float = DEFAULT_LAMBDA,
        mc_dropout_passes: int = DEFAULT_MC_DROPOUT_PASSES,
        rule_weights: list[float] | None = None,
    ):
        self.message_encoder = message_encoder
        self.conversation_encoder = conversation_encoder
        self.cyberbullying_head = cyberbullying_head
        self.grooming_head = grooming_head
        self.emotion_classifier = emotion_classifier
        self.emotion_score_head = emotion_score_head
        self.risk_fusion = risk_fusion
        self.historical_state_updater = historical_state_updater
        self.early_warning_tracker = early_warning_tracker or EarlyWarningTracker()
        self.safety_feature_extractor = safety_feature_extractor or SafetyFeatureExtractor()
        self.dependency_extractor = dependency_extractor or EmotionalDependencyExtractor()
        self.lam = lam
        self.mc_dropout_passes = mc_dropout_passes
        self.rule_weights = rule_weights

        # Modules whose stochastic forward passes feed MC Dropout (Section
        # 13): the shared encoder, conversation encoder, the two trainable
        # task heads, and the fusion stage. The frozen GoEmotions
        # classifier and the Emotion Score head are deliberately excluded.
        self._safety_modules = (
            self.message_encoder,
            self.conversation_encoder,
            self.cyberbullying_head,
            self.grooming_head,
            self.risk_fusion,
        )
        self._all_modules = self._safety_modules + (self.emotion_classifier, self.emotion_score_head)

    @classmethod
    def from_pretrained(
        cls,
        encoder_model_name: str = DEFAULT_ENCODER_NAME,
        emotion_model_name: str = DEFAULT_GOEMOTIONS_MODEL,
        d_z: int | None = None,
        persistence_window: int = DEFAULT_PERSISTENCE_WINDOW,
        mc_dropout_passes: int = DEFAULT_MC_DROPOUT_PASSES,
        early_warning_thresholds: EarlyWarningThresholds = DEFAULT_THRESHOLDS,
    ) -> "IntegratedInferencePipeline":
        """Builds every component from pretrained checkpoints with
        mutually consistent dimensions. Requires network access the first
        time `encoder_model_name` / `emotion_model_name` are downloaded.
        """
        message_encoder = MessageEncoder(model_name=encoder_model_name)
        d_z = d_z or message_encoder.d

        return cls(
            message_encoder=message_encoder,
            conversation_encoder=ConversationEncoder(d=d_z),
            cyberbullying_head=CyberbullyingHead(d=d_z, d_z=d_z),
            grooming_head=GroomingHead(d_z=d_z),
            emotion_classifier=GoEmotionsClassifier(model_name=emotion_model_name),
            emotion_score_head=EmotionScoreHead(),
            risk_fusion=PrototypeRiskFusion(),
            historical_state_updater=HistoricalStateUpdater(persistence_window=persistence_window),
            early_warning_tracker=EarlyWarningTracker(thresholds=early_warning_thresholds),
            mc_dropout_passes=mc_dropout_passes,
        )

    @torch.no_grad()
    def process(self, window: ConversationWindow) -> IntegratedInferenceResult:
        for module in self._all_modules:
            module.eval()  # guarantee steps 1-9 below are deterministic

        # Steps 1-2: rolling window (caller-built) + structured safety signals
        rule_evidence_obj = self.safety_feature_extractor.rule_extractor.extract_evidence(window)
        features = self.safety_feature_extractor.extract(window)
        safety_tensor = torch.tensor(features.to_vector(), dtype=torch.float32)

        # Step 3: shared message + conversation encoding
        h = self.message_encoder.encode_window(window)
        z, alpha = self.conversation_encoder.encode(h)

        # Step 4: Emotion Branch -- independent of z_t / F_t^safe, runs
        # sequentially here but has no data dependency on steps 3/5-9
        g_i = self.emotion_classifier.encode_window(window)
        g_t = max_mean_top3(g_i.transpose(0, 1))
        d_t = self.dependency_extractor.extract(window)
        m_t = map_emotions(g_t, self.emotion_classifier.label_to_index, d_t, lam=self.lam)
        s_emotion = self.emotion_score_head(m_t)

        # Step 5: Cyberbullying
        p_cb = self.cyberbullying_head.forward_stage2(h, z)
        per_message_risk = self.cyberbullying_head.risk(p_cb)
        s_cb = self.cyberbullying_head.window_score(per_message_risk)

        # Step 6: Grooming (B_t is not surfaced -- Behavior Head is untrained, Section 6)
        s_g, _b_t_unused = self.grooming_head(z, safety_tensor)

        # Steps 7-8: advance H_t and evaluate the persistence-based warning
        new_state = self.historical_state_updater.update(
            s_g=s_g, s_cb=s_cb, b_t=torch.zeros(6)
        )
        warning_triggered = self.early_warning_tracker.update(new_state)
        early_warning = EarlyWarningInfo(
            triggered=warning_triggered,
            method=EARLY_WARNING_METHOD,
            accumulated_risk=new_state.accumulated_risk,
            risk_trend=new_state.risk_trend,
            risk_trend_label=trend_label(new_state.risk_trend),
            persistence=new_state.persistence,
        )

        # Step 9: rule score + prototype two-stage fusion (no S_e, Section 11)
        s_r_tilde = torch.tensor(rule_safety_score(features.rule_signals, weights=self.rule_weights))
        fused = self.risk_fusion(s_cb, s_g, s_r_tilde, s_emotion)

        # Step 10: MC Dropout uncertainty around the deterministic estimate above
        # (illustrative only -- see LIMITATIONS)
        uncertainty_estimate = self._mc_dropout(window, safety_tensor, s_r_tilde, s_emotion)

        # Step 11: evidence + the currently-operable Human Review condition,
        # combined with the early-warning condition (Section 11/13/19.5)
        evidence = extract_evidence(
            per_message_risk,
            alpha,
            rule_evidence_obj,
            g_i,
            self.emotion_classifier.label_to_index,
            d_t,
            self.emotion_score_head,
            lam=self.lam,
        )
        review_flag = currently_operable_review(
            s_r_tilde=s_r_tilde,
            q_threat_phrase=features.rule_signals.threat_phrase,
        )
        human_review_flag = bool(review_flag.item()) or warning_triggered

        # Step 12: dashboard-ready bundle
        return IntegratedInferenceResult(
            safety_score=fused.safety_score,
            emotion_score=fused.emotion_score,
            mapped_emotions=m_t,
            overall_score=fused.overall_score,
            component_scores={
                "cyberbullying": s_cb,
                "grooming": s_g,
                "rule_score": s_r_tilde,
            },
            historical_state=new_state,
            risk_trend_label=trend_label(new_state.risk_trend),
            early_warning=early_warning,
            evidence=evidence,
            uncertainty_estimate=uncertainty_estimate,
            human_review_required=human_review_flag,
            limitations=LIMITATIONS,
        )

    def _mc_dropout(
        self,
        window: ConversationWindow,
        safety_tensor: torch.Tensor,
        s_r_tilde: torch.Tensor,
        s_emotion: torch.Tensor,
    ) -> UncertaintyEstimate:
        """Re-runs the shared encoder and the trainable heads + fusion N
        times with MC Dropout enabled, holding F_t^safe, S~_r(t), and
        S_emotion(t) fixed -- this is what Section 13 means when it says
        the estimate captures uncertainty in the trainable safety and
        fusion components, not in the frozen emotion branch. Illustrative
        only until those components are trained/calibrated (Section
        19.5)."""

        def predict_fn() -> torch.Tensor:
            h = self.message_encoder.encode_window(window)
            z, _ = self.conversation_encoder.encode(h)
            p_cb = self.cyberbullying_head.forward_stage2(h, z)
            risk = self.cyberbullying_head.risk(p_cb)
            s_cb = self.cyberbullying_head.window_score(risk)
            s_g, _ = self.grooming_head(z, safety_tensor)
            fused = self.risk_fusion(s_cb, s_g, s_r_tilde, s_emotion)
            return fused.overall_score

        for module in self._safety_modules:
            enable_mc_dropout(module)
        try:
            return mc_dropout_stats(predict_fn, n=self.mc_dropout_passes)
        finally:
            for module in self._safety_modules:
                module.eval()
