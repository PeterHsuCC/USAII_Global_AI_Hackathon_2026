"""Builds the shared, expensive model components once at process startup.

Two modes (`settings.model_runtime_mode`):

- "stub": tiny, randomly-initialized encoders + zero-signal LLM/dependency
  stand-ins. No network, fast, deterministic. Used by the automated test
  suite and local dev iteration -- mirrors the pattern in
  tests/model/_tiny_bert.py / _tiny_emotion_classifier.py (duplicated here
  rather than imported, since backend/ shouldn't depend on tests/).
- "real": loads the message_encoder.pt + cyberbullying_head.pt checkpoints
  (trained together by scripts/train_cyberbullying.py) for the
  cyberbullying signal, the grooming_*_B.pt trio (trained together by
  scripts/train_grooming.py) for the grooming signal, plus the real
  LLM-based safety/dependency extractors (require ANTHROPIC_API_KEY).

Grooming checkpoint wiring, confirmed by reading grooming_head.py and
train_grooming.py rather than assumed: GroomingHead bakes `safety_dim`
into its first Linear layer's input width at construction time, and each
variant was fine-tuned end-to-end together with its OWN message_encoder +
conversation_encoder (not the one shared with cyberbullying). Variant B
(text + rule signals, safety_dim=5) is loaded here -- it scored marginally
above Variant A on the official PAN12 test (precision=0.790/recall=0.763
vs 0.781/0.744, Section 6/19.1) and, unlike Variant C, needs no
ANTHROPIC_API_KEY at training time. IntegratedInferencePipeline takes the
dedicated encoder pair via `grooming_message_encoder`/
`grooming_conversation_encoder` and slices the live safety-feature vector
down to GroomingHead.safety_dim itself (see its _grooming_safety_tensor),
so this loader only needs to build+load the matching checkpoint trio.
Variant C (LLM-augmented) has only been smoke-tested on a 300-conversation
PAN12 subset (Section 6), not the full corpus -- that checkpoint exists on
disk (trained_weights/grooming_*_C.pt) but isn't production-grade, so it
is deliberately not loaded here; REAL_MODE_EXTRA_LIMITATION below surfaces
that gap explicitly.

HistoricalStateUpdater/EarlyWarningTracker are deliberately NOT built here:
they are per-conversation state, and a global instance would leak risk
history between unrelated cases (see plan's "Important correctness note").
backend.model_runtime.job_runner builds a fresh pair per job and wraps
them around the shared heavy modules assembled here.
"""

import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizer

from backend.config import settings
from risk_detection.model import (
    ConversationEncoder,
    CyberbullyingHead,
    EmotionScoreHead,
    GoEmotionsClassifier,
    GroomingHead,
    MessageEncoder,
    PrototypeRiskFusion,
)
from risk_detection.signals.rules import RuleSignalExtractor
from risk_detection.signals.safety_features import SafetyFeatureExtractor

MODEL_VERSION_STUB = "integrated-pipeline-stub-v1"
MODEL_VERSION_REAL = "integrated-pipeline-real-v1"
RULE_VERSION = "rules-v1"
PREPROCESSING_VERSION = "preprocessing-v1"

REAL_MODE_EXTRA_LIMITATION = (
    "Grooming score uses Variant B's trained checkpoint (text + rule-based "
    "safety signals only, safety_dim=5): the LLM-augmented Variant C has only "
    "been smoke-tested on a 300-conversation PAN12 subset (Section 6), not the "
    "full corpus, so its checkpoint is not production-grade and is not loaded "
    "here; the underlying label is also PAN12's predator-identity-derived weak "
    "label, not a direct grooming annotation (Section 6)."
)


@dataclass(frozen=True)
class ModelComponents:
    message_encoder: MessageEncoder
    conversation_encoder: ConversationEncoder
    cyberbullying_head: CyberbullyingHead
    grooming_head: GroomingHead
    emotion_classifier: GoEmotionsClassifier
    emotion_score_head: EmotionScoreHead
    risk_fusion: PrototypeRiskFusion
    safety_feature_extractor: SafetyFeatureExtractor
    dependency_extractor: object
    model_version: str
    extra_limitations: tuple[str, ...] = ()
    # None in stub mode -- IntegratedInferencePipeline falls back to the
    # shared message_encoder/conversation_encoder above when these are None.
    grooming_message_encoder: MessageEncoder | None = None
    grooming_conversation_encoder: ConversationEncoder | None = None


class _ZeroLLMSafetyExtractor:
    """Stub stand-in for the real (Claude-backed) LLMSafetySignalExtractor."""

    def extract(self, window):
        from risk_detection import LLMSafetySignals

        return LLMSafetySignals(
            secrecy=0.0, isolation=0.0, dependency=0.0, sexual_escalation=0.0, threat=0.0, coercion=0.0
        )


class _ZeroDependencyExtractor:
    def extract(self, window) -> float:
        return 0.0


_STUB_VOCAB = (
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "hello",
    "world",
    "user",
    "you",
    "are",
    "secret",
    "stop",
    "test",
    "a",
    "b",
    ":",
)

_STUB_EMOTION_LABELS = ("admiration", "fear", "nervousness", "grief", "sadness", "anger", "caring", "love", "neutral")


def _build_stub_bert(hidden_size: int) -> tuple[BertTokenizer, BertModel]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(_STUB_VOCAB))
        vocab_path = f.name

    tokenizer = BertTokenizer(vocab_file=vocab_path)
    config = BertConfig(
        vocab_size=len(_STUB_VOCAB),
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=16,
        max_position_embeddings=32,
    )
    model = BertModel(config)
    model.eval()
    return tokenizer, model


def _build_stub_emotion_classifier(hidden_size: int) -> tuple[BertTokenizer, BertForSequenceClassification]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(_STUB_VOCAB))
        vocab_path = f.name

    tokenizer = BertTokenizer(vocab_file=vocab_path)
    config = BertConfig(
        vocab_size=len(_STUB_VOCAB),
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=16,
        max_position_embeddings=32,
        num_labels=len(_STUB_EMOTION_LABELS),
        id2label={i: label for i, label in enumerate(_STUB_EMOTION_LABELS)},
        label2id={label: i for i, label in enumerate(_STUB_EMOTION_LABELS)},
        problem_type="multi_label_classification",
    )
    model = BertForSequenceClassification(config)
    model.eval()
    return tokenizer, model


def _load_stub(hidden_size: int = 8) -> ModelComponents:
    tokenizer, bert = _build_stub_bert(hidden_size)
    emo_tokenizer, emo_model = _build_stub_emotion_classifier(hidden_size)

    return ModelComponents(
        message_encoder=MessageEncoder(tokenizer=tokenizer, encoder=bert),
        conversation_encoder=ConversationEncoder(d=hidden_size),
        cyberbullying_head=CyberbullyingHead(d=hidden_size, d_z=hidden_size),
        grooming_head=GroomingHead(d_z=hidden_size, safety_dim=11),
        emotion_classifier=GoEmotionsClassifier(tokenizer=emo_tokenizer, encoder=emo_model),
        emotion_score_head=EmotionScoreHead(),
        risk_fusion=PrototypeRiskFusion(),
        safety_feature_extractor=SafetyFeatureExtractor(
            llm_extractor=_ZeroLLMSafetyExtractor(),
            rule_extractor=RuleSignalExtractor(),
        ),
        dependency_extractor=_ZeroDependencyExtractor(),
        model_version=MODEL_VERSION_STUB,
    )


def _load_real(weights_dir: Path) -> ModelComponents:
    import json

    from risk_detection.signals.emotional_dependency import EmotionalDependencyExtractor
    from risk_detection.signals.llm_safety import LLMSafetySignalExtractor

    # cyberbullying_head.pt was trained against the 6-class label set in
    # label_mapping.json (scripts/train_cyberbullying.py builds num_classes
    # and non_bullying_index from that same mapping) -- CyberbullyingHead's
    # own defaults (num_classes=2) don't match and would shape-mismatch on
    # load_state_dict, so read the real mapping rather than hardcode it.
    label_mapping: dict[str, int] = json.loads((weights_dir / "label_mapping.json").read_text())
    num_classes = len(label_mapping)
    non_bullying_index = label_mapping["not_cyberbullying"]

    # message_encoder.pt / cyberbullying_head.pt were trained together by
    # scripts/train_cyberbullying.py, whose --model-name defaults to
    # distilbert-base-uncased -- must match here or load_state_dict will
    # shape-mismatch.
    message_encoder = MessageEncoder(model_name="distilbert-base-uncased")
    conversation_encoder = ConversationEncoder(d=message_encoder.d)
    cyberbullying_head = CyberbullyingHead(
        d=message_encoder.d,
        d_z=message_encoder.d,
        num_classes=num_classes,
        non_bullying_index=non_bullying_index,
    )

    message_encoder.load_state_dict(torch.load(weights_dir / "message_encoder.pt", map_location="cpu"))
    cyberbullying_head.load_state_dict(torch.load(weights_dir / "cyberbullying_head.pt", map_location="cpu"))

    message_encoder.eval()
    cyberbullying_head.eval()

    # Grooming: Variant B's own encoder pair + head, fine-tuned together by
    # scripts/train_grooming.py (SAFETY_DIM_BY_VARIANT["B"] == 5, rule
    # signals only) -- see the module docstring for why this can't reuse
    # the cyberbullying message_encoder/conversation_encoder above.
    grooming_message_encoder = MessageEncoder(model_name="distilbert-base-uncased")
    grooming_conversation_encoder = ConversationEncoder(d=grooming_message_encoder.d)
    grooming_head = GroomingHead(d_z=grooming_message_encoder.d, safety_dim=5)

    grooming_message_encoder.load_state_dict(
        torch.load(weights_dir / "grooming_message_encoder_B.pt", map_location="cpu")
    )
    grooming_conversation_encoder.load_state_dict(
        torch.load(weights_dir / "grooming_conversation_encoder_B.pt", map_location="cpu")
    )
    grooming_head.load_state_dict(torch.load(weights_dir / "grooming_head_B.pt", map_location="cpu"))

    grooming_message_encoder.eval()
    grooming_conversation_encoder.eval()
    grooming_head.eval()

    emotion_classifier = GoEmotionsClassifier()
    emotion_score_head = EmotionScoreHead()
    risk_fusion = PrototypeRiskFusion()

    return ModelComponents(
        message_encoder=message_encoder,
        conversation_encoder=conversation_encoder,
        cyberbullying_head=cyberbullying_head,
        grooming_head=grooming_head,
        emotion_classifier=emotion_classifier,
        emotion_score_head=emotion_score_head,
        risk_fusion=risk_fusion,
        safety_feature_extractor=SafetyFeatureExtractor(
            llm_extractor=LLMSafetySignalExtractor(),
            rule_extractor=RuleSignalExtractor(),
        ),
        dependency_extractor=EmotionalDependencyExtractor(),
        model_version=MODEL_VERSION_REAL,
        extra_limitations=(REAL_MODE_EXTRA_LIMITATION,),
        grooming_message_encoder=grooming_message_encoder,
        grooming_conversation_encoder=grooming_conversation_encoder,
    )


_lock = threading.Lock()
_components: ModelComponents | None = None


def get_model_components() -> ModelComponents:
    global _components
    with _lock:
        if _components is None:
            if settings.model_runtime_mode == "real":
                _components = _load_real(settings.trained_weights_dir)
            else:
                _components = _load_stub()
        return _components


def current_model_version() -> str:
    """The model_version a job submitted right now will be stamped with --
    cheap (no model loading) so the synchronous case-submission request
    doesn't trigger loading the heavy components; only the async worker
    actually needs them loaded.
    """
    return MODEL_VERSION_REAL if settings.model_runtime_mode == "real" else MODEL_VERSION_STUB


def reset_model_components() -> None:
    """Test-only hook to force a reload (e.g. after monkeypatching mode)."""
    global _components
    with _lock:
        _components = None