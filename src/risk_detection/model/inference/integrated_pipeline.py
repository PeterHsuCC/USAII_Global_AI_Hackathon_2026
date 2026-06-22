from dataclasses import dataclass

import torch

from ...conversation import ConversationWindow
from ...signals.emotional_dependency import EmotionalDependencyExtractor
from ...signals.llm_safety import LLMRefusalError
from ...signals.rule_score import rule_safety_score
from ...signals.safety_features import SafetyFeatureExtractor, SafetyFeatures
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
    "illustrative: EmotionScoreHead and RiskFusion have not yet been trained "
    "or calibrated. The Grooming Head is only as good as whatever weights it "
    "was constructed with -- untrained/random unless the caller explicitly "
    "wired in a trained checkpoint (Section 19.5). Only the rule-based and "
    "LLM-signal-based human review conditions and the persistence-based "
    "early warning below are currently operable.",
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
        grooming_message_encoder: MessageEncoder | None = None,
        grooming_conversation_encoder: ConversationEncoder | None = None,
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
        # scripts/train_grooming.py fine-tunes message_encoder +
        # conversation_encoder jointly with each GroomingHead variant, so a
        # trained grooming checkpoint expects ITS OWN encoder pair, not the
        # one shared with cyberbullying -- these default to the shared pair
        # only when the caller has no variant-specific encoders to give
        # (stub/test mode, from_pretrained()).
        self.grooming_message_encoder = grooming_message_encoder or message_encoder
        self.grooming_conversation_encoder = grooming_conversation_encoder or conversation_encoder
        self.lam = lam
        self.mc_dropout_passes = mc_dropout_passes
        self.rule_weights = rule_weights

        # Modules whose stochastic forward passes feed MC Dropout (Section
        # 13): both encoder pairs, the two trainable task heads, and the
        # fusion stage. The frozen GoEmotions classifier and the Emotion
        # Score head are deliberately excluded.
        self._safety_modules = (
            self.message_encoder,
            self.conversation_encoder,
            self.grooming_message_encoder,
            self.grooming_conversation_encoder,
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
        """Convenience constructor for ad-hoc/exploratory use: downloads the
        pretrained encoder backbones (`encoder_model_name`,
        `emotion_model_name`) but builds every trainable task head
        (CyberbullyingHead, GroomingHead, EmotionScoreHead, RiskFusion)
        freshly initialized -- it does NOT load this project's own trained
        checkpoints. For that, see backend.model_runtime.loader._load_real,
        which loads message_encoder.pt/cyberbullying_head.pt and the
        Variant B grooming_*.pt trio (Section 19.5). Requires network
        access the first time `encoder_model_name` / `emotion_model_name`
        are downloaded.
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
        grooming_safety_tensor = self._grooming_safety_tensor(features)

        # Step 3: shared message + conversation encoding
        h = self.message_encoder.encode_window(window)
        z, alpha = self.conversation_encoder.encode(h)
        z_g = self._grooming_z(window, z)

        # Step 4: Emotion Branch -- independent of z_t / F_t^safe, runs
        # sequentially here but has no data dependency on steps 3/5-9
        g_i = self.emotion_classifier.encode_window(window)
        g_t = max_mean_top3(g_i.transpose(0, 1))
        try:
            d_t = self.dependency_extractor.extract(window)
        except LLMRefusalError as e:
            print(f"  LLM refused emotional-dependency extraction (category={e.category}); using 0.0")
            d_t = 0.0
        m_t = map_emotions(g_t, self.emotion_classifier.label_to_index, d_t, lam=self.lam)
        s_emotion = self.emotion_score_head(m_t)

        # Step 5: Cyberbullying
        p_cb = self.cyberbullying_head.forward_stage2(h, z)
        per_message_risk = self.cyberbullying_head.risk(p_cb)
        s_cb = self.cyberbullying_head.window_score(per_message_risk)

        # Step 6: Grooming (B_t is not surfaced -- Behavior Head is untrained, Section 6)
        s_g, _b_t_unused = self.grooming_head(z_g, grooming_safety_tensor)

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
        uncertainty_estimate = self._mc_dropout(window, grooming_safety_tensor, s_r_tilde, s_emotion)

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
            llm_signal_max=max(features.llm_signals.to_vector()),
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
        grooming_safety_tensor: torch.Tensor,
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
            z_g = self._grooming_z(window, z)
            s_g, _ = self.grooming_head(z_g, grooming_safety_tensor)
            fused = self.risk_fusion(s_cb, s_g, s_r_tilde, s_emotion)
            return fused.overall_score

        for module in self._safety_modules:
            enable_mc_dropout(module)
        try:
            return mc_dropout_stats(predict_fn, n=self.mc_dropout_passes)
        finally:
            for module in self._safety_modules:
                module.eval()

    def _grooming_z(self, window: ConversationWindow, shared_z: torch.Tensor) -> torch.Tensor:
        """Variant B's GroomingHead was fine-tuned jointly with its own
        message/conversation encoder (scripts/train_grooming.py), not the
        cyberbullying-shared one -- re-encode only when a distinct
        grooming encoder pair was actually supplied."""
        if self.grooming_message_encoder is self.message_encoder and (
            self.grooming_conversation_encoder is self.conversation_encoder
        ):
            return shared_z
        h_g = self.grooming_message_encoder.encode_window(window)
        z_g, _alpha_g = self.grooming_conversation_encoder.encode(h_g)
        return z_g

    def _grooming_safety_tensor(self, features: SafetyFeatures) -> torch.Tensor:
        """Builds the safety-feature vector GroomingHead actually expects,
        matching scripts/train_grooming.py's build_safety_features:
        safety_dim=0 -> none (Variant A), safety_dim=len(rule vector) ->
        rule signals only (Variant B), safety_dim=len(full vector) ->
        LLM + rule signals (Variant C). GroomingHead bakes safety_dim into
        its first Linear layer's input width at construction time, so
        feeding it the wrong width would shape-mismatch the forward pass."""
        safety_dim = self.grooming_head.safety_dim
        rule_vector = features.rule_signals.to_vector()
        full_vector = features.to_vector()
        if safety_dim == 0:
            return torch.zeros(0)
        if safety_dim == len(rule_vector):
            return torch.tensor(rule_vector, dtype=torch.float32)
        if safety_dim == len(full_vector):
            return torch.tensor(full_vector, dtype=torch.float32)
        raise ValueError(
            f"GroomingHead.safety_dim={safety_dim} matches neither rule-only "
            f"({len(rule_vector)}) nor full ({len(full_vector)}) safety feature width"
        )
