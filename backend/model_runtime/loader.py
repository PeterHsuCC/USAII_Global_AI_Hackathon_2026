"""Builds the shared, expensive model components once at process startup.

Three modes (`settings.model_runtime_mode`):

- "stub": tiny, randomly-initialized encoders + zero-signal LLM/dependency
  stand-ins. No network, fast, deterministic. Used by the automated test
  suite and local dev iteration -- mirrors the pattern in
  tests/model/_tiny_bert.py / _tiny_emotion_classifier.py (duplicated here
  rather than imported, since backend/ shouldn't depend on tests/). Note
  this also means cyberbullying_head/grooming_head are random in this
  mode, not just the encoders -- neither is loaded from a checkpoint, so
  stub-mode cyberbullying/grooming component scores carry no real signal
  either, found via a live test where escalating a conversation's final
  message from small talk to an explicit threat barely moved either score.
- "local": the real trained_weights/ checkpoints for cyberbullying and
  grooming (same loading code as "real" below) plus the real pretrained
  GoEmotions classifier, but the zero-signal LLM/dependency stand-ins
  instead of the Claude-backed ones -- meaningful cyberbullying/grooming/
  emotion-mapping scores with no ANTHROPIC_API_KEY and no per-case cost,
  for free local UI testing. LLM-sourced signals (secrecy, isolation, ...)
  and the emotional-dependency proxy stay at 0, same as "stub".
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
MODEL_VERSION_LOCAL = "integrated-pipeline-local-v1"
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

LOCAL_MODE_EXTRA_LIMITATION = (
    "This case was analyzed in 'local' mode: cyberbullying/grooming/emotion "
    "scores use the real trained checkpoints, but no LLM call was made -- "
    "every LLM-sourced safety signal (secrecy, isolation, dependency, "
    "sexual_escalation, threat, coercion) and the emotional-dependency proxy "
    "are fixed at 0, the same as stub mode, not a genuine absence of those "
    "signals in the conversation."
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

# Must match the max_length passed to MessageEncoder/GoEmotionsClassifier in
# _load_stub below -- otherwise a message that tokenizes past this many
# tokens overflows the tiny model's position-embedding table instead of
# being truncated first, e.g. "size of tensor a (97) must match the size
# of tensor b (32)".
_STUB_MAX_POSITION_EMBEDDINGS = 32


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
        max_position_embeddings=_STUB_MAX_POSITION_EMBEDDINGS,
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
        max_position_embeddings=_STUB_MAX_POSITION_EMBEDDINGS,
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
        message_encoder=MessageEncoder(
            tokenizer=tokenizer, encoder=bert, max_length=_STUB_MAX_POSITION_EMBEDDINGS
        ),
        conversation_encoder=ConversationEncoder(d=hidden_size),
        cyberbullying_head=CyberbullyingHead(d=hidden_size, d_z=hidden_size),
        grooming_head=GroomingHead(d_z=hidden_size, safety_dim=11),
        emotion_classifier=GoEmotionsClassifier(
            tokenizer=emo_tokenizer, encoder=emo_model, max_length=_STUB_MAX_POSITION_EMBEDDINGS
        ),
        emotion_score_head=EmotionScoreHead(),
        risk_fusion=PrototypeRiskFusion(),
        safety_feature_extractor=SafetyFeatureExtractor(
            llm_extractor=_ZeroLLMSafetyExtractor(),
            rule_extractor=RuleSignalExtractor(),
        ),
        dependency_extractor=_ZeroDependencyExtractor(),
        model_version=MODEL_VERSION_STUB,
    )


def _load_real(weights_dir: Path, *, use_real_llm_extractors: bool = True) -> ModelComponents:
    import json

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

    if use_real_llm_extractors:
        from risk_detection.signals.emotional_dependency import EmotionalDependencyExtractor
        from risk_detection.signals.llm_safety import LLMSafetySignalExtractor

        llm_extractor = LLMSafetySignalExtractor()
        dependency_extractor = EmotionalDependencyExtractor()
        model_version = MODEL_VERSION_REAL
        extra_limitations = (REAL_MODE_EXTRA_LIMITATION,)
    else:
        llm_extractor = _ZeroLLMSafetyExtractor()
        dependency_extractor = _ZeroDependencyExtractor()
        model_version = MODEL_VERSION_LOCAL
        extra_limitations = (REAL_MODE_EXTRA_LIMITATION, LOCAL_MODE_EXTRA_LIMITATION)

    return ModelComponents(
        message_encoder=message_encoder,
        conversation_encoder=conversation_encoder,
        cyberbullying_head=cyberbullying_head,
        grooming_head=grooming_head,
        emotion_classifier=emotion_classifier,
        emotion_score_head=emotion_score_head,
        risk_fusion=risk_fusion,
        safety_feature_extractor=SafetyFeatureExtractor(
            llm_extractor=llm_extractor,
            rule_extractor=RuleSignalExtractor(),
        ),
        dependency_extractor=dependency_extractor,
        model_version=model_version,
        extra_limitations=extra_limitations,
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
            elif settings.model_runtime_mode == "local":
                _components = _load_real(settings.trained_weights_dir, use_real_llm_extractors=False)
            else:
                _components = _load_stub()
        return _components


def current_model_version() -> str:
    """The model_version a job submitted right now will be stamped with --
    cheap (no model loading) so the synchronous case-submission request
    doesn't trigger loading the heavy components; only the async worker
    actually needs them loaded.
    """
    if settings.model_runtime_mode == "real":
        return MODEL_VERSION_REAL
    if settings.model_runtime_mode == "local":
        return MODEL_VERSION_LOCAL
    return MODEL_VERSION_STUB


def reset_model_components() -> None:
    """Test-only hook to force a reload (e.g. after monkeypatching mode)."""
    global _components
    with _lock:
        _components = None