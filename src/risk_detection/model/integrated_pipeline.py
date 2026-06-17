from dataclasses import dataclass

import torch

from ..conversation import ConversationWindow
from ..signals.emotional_dependency import EmotionalDependencyExtractor
from ..signals.rule_score import rule_safety_score
from ..signals.safety_features import SafetyFeatureExtractor
from .aggregation import max_mean_top3
from .conversation_encoder import ConversationEncoder
from .cyberbullying_head import CyberbullyingHead
from .early_detection_head import EarlyDetectionHead
from .emotion_classifier import DEFAULT_GOEMOTIONS_MODEL, GoEmotionsClassifier
from .emotion_mapping import DEFAULT_LAMBDA, map_emotions
from .emotion_score_head import EmotionScoreHead
from .evidence import EvidenceBundle, extract_evidence
from .grooming_head import GroomingHead
from .historical_state import (
    DEFAULT_PERSISTENCE_WINDOW,
    HistoricalRiskState,
    HistoricalStateUpdater,
    trend_label,
)
from .message_encoder import DEFAULT_ENCODER_NAME, MessageEncoder
from .risk_fusion import RiskFusion
from .uncertainty import (
    DEFAULT_MC_DROPOUT_PASSES,
    UncertaintyEstimate,
    enable_mc_dropout,
    human_review_required,
    mc_dropout_stats,
)

LIMITATIONS = (
    "This is a decision-support system. It does not automatically label a "
    "person as an offender or victim, make a diagnosis, report a case, or "
    "trigger enforcement action.",
    "Attention-based evidence indicates model focus, not causal or legal proof.",
    "Confidence represents prediction stability, not correctness probability.",
    "The MC Dropout uncertainty estimate primarily captures uncertainty from "
    "the trainable safety and fusion components; it does not fully represent "
    "uncertainty in the frozen GoEmotions classifier or the emotion mapping layer.",
    "All evidence must be interpreted by a human analyst.",
)


@dataclass
class IntegratedInferenceResult:
    """Section 15 step 12: everything the dashboard needs for one
    Conversation Window / time step."""

    safety_score: torch.Tensor  # S_safety(t)
    emotion_score: torch.Tensor  # S_emotion(t)
    overall_score: torch.Tensor  # R_t, deterministic forward pass
    component_scores: dict[str, torch.Tensor]  # cyberbullying, grooming, early_predator, rule_score
    historical_state: HistoricalRiskState  # H_t, after this step's update
    risk_trend_label: str
    evidence: EvidenceBundle
    uncertainty_estimate: UncertaintyEstimate  # R_hat_t, variance, uncertainty, confidence
    human_review_required: bool
    limitations: tuple[str, ...]


class IntegratedInferencePipeline:
    """Section 15: Stage 2 Integrated Inference Flow. Wires together every
    previously-built component into one pass over a rolling Conversation
    Window, computing each expensive step exactly once -- the LLM safety
    signal call, the shared message/conversation encoder, the GoEmotions
    classifier -- rather than letting the per-task pipelines from earlier
    sections each repeat them independently.

    `historical_state_updater` is stateful and persists across successive
    `process()` calls on the same conversation: H_{t-1} is read before the
    Early Detection Head runs, and only advanced to H_t afterwards
    (Section 9's one-step delay).
    """

    def __init__(
        self,
        message_encoder: MessageEncoder,
        conversation_encoder: ConversationEncoder,
        cyberbullying_head: CyberbullyingHead,
        grooming_head: GroomingHead,
        early_detection_head: EarlyDetectionHead,
        emotion_classifier: GoEmotionsClassifier,
        emotion_score_head: EmotionScoreHead,
        risk_fusion: RiskFusion,
        historical_state_updater: HistoricalStateUpdater,
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
        self.early_detection_head = early_detection_head
        self.emotion_classifier = emotion_classifier
        self.emotion_score_head = emotion_score_head
        self.risk_fusion = risk_fusion
        self.historical_state_updater = historical_state_updater
        self.safety_feature_extractor = safety_feature_extractor or SafetyFeatureExtractor()
        self.dependency_extractor = dependency_extractor or EmotionalDependencyExtractor()
        self.lam = lam
        self.mc_dropout_passes = mc_dropout_passes
        self.rule_weights = rule_weights

        # Modules whose stochastic forward passes feed MC Dropout (Section
        # 13): the shared encoder, conversation encoder, the three task
        # heads, and the fusion stage. The frozen GoEmotions classifier and
        # the Emotion Score head are deliberately excluded.
        self._safety_modules = (
            self.message_encoder,
            self.conversation_encoder,
            self.cyberbullying_head,
            self.grooming_head,
            self.early_detection_head,
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
            early_detection_head=EarlyDetectionHead(d_z=d_z),
            emotion_classifier=GoEmotionsClassifier(model_name=emotion_model_name),
            emotion_score_head=EmotionScoreHead(),
            risk_fusion=RiskFusion(),
            historical_state_updater=HistoricalStateUpdater(persistence_window=persistence_window),
            mc_dropout_passes=mc_dropout_passes,
        )

    @torch.no_grad()
    def process(self, window: ConversationWindow) -> IntegratedInferenceResult:
        for module in self._all_modules:
            module.eval()  # guarantee steps 1-9 below are deterministic

        h_prev = self.historical_state_updater.state  # H_{t-1}, read before any update

        # Steps 1-2: rolling window (caller-built) + structured safety signals
        rule_evidence_obj = self.safety_feature_extractor.rule_extractor.extract_evidence(window)
        features = self.safety_feature_extractor.extract(window)
        safety_tensor = torch.tensor(features.to_vector(), dtype=torch.float32)

        # Step 3: shared message + conversation encoding
        h = self.message_encoder.encode_window(window)
        z, alpha = self.conversation_encoder.encode(h)

        # Step 4: Emotion Branch -- independent of z_t / F_t^safe, runs
        # sequentially here but has no data dependency on steps 3/5-7
        g_i = self.emotion_classifier.encode_window(window)
        g_t = max_mean_top3(g_i.transpose(0, 1))
        d_t = self.dependency_extractor.extract(window)
        m_t = map_emotions(g_t, self.emotion_classifier.label_to_index, d_t, lam=self.lam)
        s_emotion = self.emotion_score_head(m_t)

        # Step 5: Cyberbullying
        p_cb = self.cyberbullying_head.forward_stage2(h, z)
        per_message_risk = self.cyberbullying_head.risk(p_cb)
        s_cb = self.cyberbullying_head.window_score(per_message_risk)

        # Step 6: Grooming
        s_g, b_t = self.grooming_head(z, safety_tensor)

        # Step 7: Early Detection, using H_{t-1}
        h_prev_tensor = h_prev.to_vector()
        s_e = self.early_detection_head(z, safety_tensor, h_prev_tensor)

        # Step 8: advance H_{t-1} -> H_t only now that S_e(t) is computed
        new_state = self.historical_state_updater.update(s_g=s_g, s_cb=s_cb, b_t=b_t)

        # Step 9: rule score + two-stage fusion
        s_r_tilde = torch.tensor(rule_safety_score(features.rule_signals, weights=self.rule_weights))
        fused = self.risk_fusion(s_cb, s_g, s_e, s_r_tilde, s_emotion)

        # Step 10: MC Dropout uncertainty around the deterministic estimate above
        uncertainty_estimate = self._mc_dropout(window, safety_tensor, h_prev_tensor, s_r_tilde, s_emotion)

        # Step 11: evidence + human review
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
        review_flag = human_review_required(
            r_hat=uncertainty_estimate.mean,
            confidence=uncertainty_estimate.confidence,
            s_r_tilde=s_r_tilde,
        )

        # Step 12: dashboard-ready bundle
        return IntegratedInferenceResult(
            safety_score=fused.safety_score,
            emotion_score=fused.emotion_score,
            overall_score=fused.overall_score,
            component_scores={
                "cyberbullying": s_cb,
                "grooming": s_g,
                "early_predator": s_e,
                "rule_score": s_r_tilde,
            },
            historical_state=new_state,
            risk_trend_label=trend_label(new_state.risk_trend),
            evidence=evidence,
            uncertainty_estimate=uncertainty_estimate,
            human_review_required=bool(review_flag.item()),
            limitations=LIMITATIONS,
        )

    def _mc_dropout(
        self,
        window: ConversationWindow,
        safety_tensor: torch.Tensor,
        h_prev_tensor: torch.Tensor,
        s_r_tilde: torch.Tensor,
        s_emotion: torch.Tensor,
    ) -> UncertaintyEstimate:
        """Re-runs the shared encoder and the three task heads + fusion N
        times with MC Dropout enabled, holding F_t^safe, H_{t-1}, S~_r(t),
        and S_emotion(t) fixed -- this is what Section 13 means when it
        says the estimate captures uncertainty in the trainable safety and
        fusion components, not in the frozen emotion branch."""

        def predict_fn() -> torch.Tensor:
            h = self.message_encoder.encode_window(window)
            z, _ = self.conversation_encoder.encode(h)
            p_cb = self.cyberbullying_head.forward_stage2(h, z)
            risk = self.cyberbullying_head.risk(p_cb)
            s_cb = self.cyberbullying_head.window_score(risk)
            s_g, _ = self.grooming_head(z, safety_tensor)
            s_e = self.early_detection_head(z, safety_tensor, h_prev_tensor)
            fused = self.risk_fusion(s_cb, s_g, s_e, s_r_tilde, s_emotion)
            return fused.overall_score

        for module in self._safety_modules:
            enable_mc_dropout(module)
        try:
            return mc_dropout_stats(predict_fn, n=self.mc_dropout_passes)
        finally:
            for module in self._safety_modules:
                module.eval()
