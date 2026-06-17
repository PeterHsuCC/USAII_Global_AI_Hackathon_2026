"""Small-scale demonstration of Section 12's training strategy.

This is NOT a real training run -- there is no labeled dataset behind it.
The six hand-crafted conversation snippets and their labels below are
purely illustrative. The goal is only to prove that the Stage 1 -> Stage 2
-> Stage 3 procedure described in the report actually runs end-to-end and
that every loss decreases, not to produce a model with any real detection
ability. Training on real data requires real labeled conversations.

Uses a tiny, randomly-initialized BERT/GoEmotions-shaped classifier built
entirely in memory (no download, no network), so the whole script runs in
a few seconds on CPU. Swap in MessageEncoder()/GoEmotionsClassifier() with
their default model names for a run against real pretrained checkpoints.
"""

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch import optim
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertModel,
    BertTokenizer,
)

from risk_detection import RuleSignals, rule_safety_score
from risk_detection.model import (
    ConversationEncoder,
    CyberbullyingHead,
    EarlyDetectionHead,
    EmotionScoreHead,
    GoEmotionsClassifier,
    GroomingHead,
    MessageEncoder,
    RiskFusion,
    behavior_loss,
    binary_review_loss,
    cyberbullying_loss,
    freeze,
    map_emotions,
    masked_multitask_loss,
    max_mean_top3,
)

HIDDEN_SIZE = 16
SAFETY_DIM = 11  # F_t^safe = [L_t (6) ; Q_t (5)]
HISTORY_DIM = 15  # H_{t-1}
EMOTION_LABELS = ("fear", "nervousness", "grief", "sadness", "anger", "caring", "love", "neutral")


@dataclass
class Example:
    speaker_id: str
    texts: list[str]
    cb_labels: list[int]  # per message: 0 = bullying, 1 = non_bullying
    # [secrecy, isolation, dependency, sexual_escalation, threat, coercion,
    #  secret_request, contact_migration, age_reference, image_request, threat_phrase]
    safety_features: list[float]
    h_prev: list[float] = field(default_factory=lambda: [0.0] * HISTORY_DIM)
    y_g: float = 0.0
    # [rapid_trust_building, secrecy, isolation, emotional_dependency, sexual_escalation, coercion]
    y_b: list[float] = field(default_factory=lambda: [0.0] * 6)
    y_e: float = 0.0
    d_t: float = 0.1  # synthetic stand-in for the LLM-extracted dependency signal
    y_emo: float | None = 0.0  # None = no emotion-review label for this example
    y_overall: float = 0.0  # synthetic "should a human reviewer flag this" label


EXAMPLES = [
    Example(
        speaker_id="a",
        texts=["you are so stupid and ugly nobody likes you"],
        cb_labels=[0],
        safety_features=[0.0] * SAFETY_DIM,
        y_emo=1.0,
        y_overall=1.0,
    ),
    Example(
        speaker_id="a",
        texts=["hey how was your day today"],
        cb_labels=[1],
        safety_features=[0.0] * SAFETY_DIM,
        y_emo=0.0,
        y_overall=0.0,
    ),
    Example(
        speaker_id="a",
        texts=[
            "you can trust me i am the only one who understands you",
            "lets keep this between us",
            "add me on snapchat",
        ],
        cb_labels=[1, 1, 1],
        safety_features=[0.8, 0.7, 0.6, 0.0, 0.0, 0.3, 1.0, 1.0, 0.0, 0.0, 0.0],
        h_prev=[0.4, 0.1, 0.3] + [0.0] * 12,
        y_g=1.0,
        y_b=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        y_e=1.0,
        d_t=0.7,
        y_emo=1.0,
        y_overall=1.0,
    ),
    Example(
        speaker_id="a",
        texts=["want to study together for the test tomorrow", "sure sounds good"],
        cb_labels=[1, 1],
        safety_features=[0.0] * SAFETY_DIM,
        y_emo=None,  # demonstrates Stage 2's emotion-mask: no review label available
        y_overall=0.0,
    ),
    Example(
        speaker_id="a",
        texts=["send the pic or i will hurt you"],
        cb_labels=[0],
        safety_features=[0.0, 0.0, 0.0, 0.6, 0.9, 0.8, 0.0, 0.0, 0.0, 1.0, 1.0],
        h_prev=[0.5, 0.2, 0.4] + [0.0] * 12,
        y_g=1.0,
        y_b=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        y_e=1.0,
        d_t=0.3,
        y_emo=1.0,
        y_overall=1.0,
    ),
    Example(
        speaker_id="a",
        texts=["good morning ready for class"],
        cb_labels=[1],
        safety_features=[0.0] * SAFETY_DIM,
        y_emo=0.0,
        y_overall=0.0,
    ),
]


def _build_vocab(texts: list[str]) -> list[str]:
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    words = sorted({w for text in texts for w in text.lower().split()})
    return special + words


def _write_vocab_file(vocab: list[str]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(vocab))
        return f.name


def make_tiny_bert(vocab: list[str], hidden_size: int = HIDDEN_SIZE):
    tokenizer = BertTokenizer(vocab_file=_write_vocab_file(vocab))
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=hidden_size * 2,
        max_position_embeddings=64,
    )
    return tokenizer, BertModel(config)


def make_tiny_emotion_classifier(vocab: list[str], hidden_size: int = HIDDEN_SIZE):
    tokenizer = BertTokenizer(vocab_file=_write_vocab_file(vocab))
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=hidden_size * 2,
        max_position_embeddings=64,
        num_labels=len(EMOTION_LABELS),
        id2label={i: label for i, label in enumerate(EMOTION_LABELS)},
        label2id={label: i for i, label in enumerate(EMOTION_LABELS)},
        problem_type="multi_label_classification",
    )
    return tokenizer, BertForSequenceClassification(config)


def flatten_cb_dataset() -> tuple[list[str], list[str], torch.Tensor]:
    speakers, texts, labels = [], [], []
    for ex in EXAMPLES:
        for text, label in zip(ex.texts, ex.cb_labels):
            speakers.append(ex.speaker_id)
            texts.append(text)
            labels.append(label)
    return speakers, texts, torch.tensor(labels)


def main() -> None:
    torch.manual_seed(0)

    vocab = _build_vocab([t for ex in EXAMPLES for t in ex.texts])
    bert_tokenizer, bert_model = make_tiny_bert(vocab)
    emo_tokenizer, emo_model = make_tiny_emotion_classifier(vocab)

    message_encoder = MessageEncoder(tokenizer=bert_tokenizer, encoder=bert_model)
    conversation_encoder = ConversationEncoder(d=HIDDEN_SIZE)
    cyberbullying_head = CyberbullyingHead(d=HIDDEN_SIZE, d_z=HIDDEN_SIZE)
    grooming_head = GroomingHead(d_z=HIDDEN_SIZE, safety_dim=SAFETY_DIM)
    early_detection_head = EarlyDetectionHead(d_z=HIDDEN_SIZE, safety_dim=SAFETY_DIM, history_dim=HISTORY_DIM)
    emotion_classifier = GoEmotionsClassifier(tokenizer=emo_tokenizer, encoder=emo_model)
    emotion_score_head = EmotionScoreHead()
    risk_fusion = RiskFusion()

    trainable_modules = [
        message_encoder,
        conversation_encoder,
        cyberbullying_head,
        grooming_head,
        early_detection_head,
        emotion_score_head,
    ]
    optimizer = optim.AdamW([p for m in trainable_modules for p in m.parameters()], lr=2e-3)

    cb_speakers, cb_texts, cb_labels = flatten_cb_dataset()

    # ---------------------------------------------------------------
    # Stage 1 (Section 12.1): each task is trained on its own loss in
    # its own phase -- nothing is combined or masked yet.
    # ---------------------------------------------------------------
    print("=== Stage 1: independent task training ===")

    print("-- L_cb (single-message training) --")
    for epoch in range(5):
        optimizer.zero_grad()
        h = message_encoder(cb_speakers, cb_texts)
        p_cb = cyberbullying_head.forward_stage1(h)
        loss = cyberbullying_loss(p_cb, cb_labels).mean()
        loss.backward()
        optimizer.step()
        print(f"  epoch {epoch}: L_cb = {loss.item():.4f}")

    print("-- L_g + L_b (grooming score + behaviors) --")
    for epoch in range(5):
        optimizer.zero_grad()
        g_losses, b_losses = [], []
        for ex in EXAMPLES:
            h = message_encoder([ex.speaker_id] * len(ex.texts), ex.texts)
            z, _ = conversation_encoder.encode(h)
            safety_tensor = torch.tensor(ex.safety_features)
            s_g, b_t = grooming_head(z, safety_tensor)
            g_losses.append(binary_review_loss(s_g, torch.tensor(ex.y_g)))
            b_losses.append(behavior_loss(b_t.unsqueeze(0), torch.tensor(ex.y_b).unsqueeze(0)).squeeze(0))
        loss = torch.stack(g_losses).mean() + torch.stack(b_losses).mean()
        loss.backward()
        optimizer.step()
        print(f"  epoch {epoch}: L_g + L_b = {loss.item():.4f}")

    print("-- L_e (early detection) --")
    for epoch in range(5):
        optimizer.zero_grad()
        e_losses = []
        for ex in EXAMPLES:
            h = message_encoder([ex.speaker_id] * len(ex.texts), ex.texts)
            z, _ = conversation_encoder.encode(h)
            safety_tensor = torch.tensor(ex.safety_features)
            h_prev_tensor = torch.tensor(ex.h_prev)
            s_e = early_detection_head(z, safety_tensor, h_prev_tensor)
            e_losses.append(binary_review_loss(s_e, torch.tensor(ex.y_e)))
        loss = torch.stack(e_losses).mean()
        loss.backward()
        optimizer.step()
        print(f"  epoch {epoch}: L_e = {loss.item():.4f}")

    print("-- L_emo (frozen GoEmotions classifier; only theta/b_m update) --")
    labeled_examples = [ex for ex in EXAMPLES if ex.y_emo is not None]
    for epoch in range(5):
        optimizer.zero_grad()
        emo_losses = []
        for ex in labeled_examples:
            g_i = emotion_classifier(ex.texts)
            g_t = max_mean_top3(g_i.transpose(0, 1))
            m_t = map_emotions(g_t, emotion_classifier.label_to_index, ex.d_t)
            s_emotion = emotion_score_head(m_t)
            emo_losses.append(binary_review_loss(s_emotion, torch.tensor(ex.y_emo)))
        loss = torch.stack(emo_losses).mean()
        loss.backward()
        optimizer.step()
        print(f"  epoch {epoch}: L_emo = {loss.item():.4f}")

    # ---------------------------------------------------------------
    # Stage 2 (Section 12.2): all tasks combined into one masked
    # multi-task loss. The fourth example has no emotion-review label,
    # so its emo mask is 0 -- it doesn't affect L_emo this epoch.
    # ---------------------------------------------------------------
    print("\n=== Stage 2: masked multi-task training ===")
    for epoch in range(5):
        optimizer.zero_grad()

        h = message_encoder(cb_speakers, cb_texts)
        p_cb = cyberbullying_head.forward_stage1(h)
        cb_losses = cyberbullying_loss(p_cb, cb_labels)
        cb_mask = torch.ones(len(cb_labels))

        g_losses, b_losses, e_losses, emo_losses, emo_mask = [], [], [], [], []
        for ex in EXAMPLES:
            hm = message_encoder([ex.speaker_id] * len(ex.texts), ex.texts)
            z, _ = conversation_encoder.encode(hm)
            safety_tensor = torch.tensor(ex.safety_features)
            h_prev_tensor = torch.tensor(ex.h_prev)

            s_g, b_t = grooming_head(z, safety_tensor)
            g_losses.append(binary_review_loss(s_g, torch.tensor(ex.y_g)))
            b_losses.append(behavior_loss(b_t.unsqueeze(0), torch.tensor(ex.y_b).unsqueeze(0)).squeeze(0))

            s_e = early_detection_head(z, safety_tensor, h_prev_tensor)
            e_losses.append(binary_review_loss(s_e, torch.tensor(ex.y_e)))

            g_i = emotion_classifier(ex.texts)
            g_t = max_mean_top3(g_i.transpose(0, 1))
            m_t = map_emotions(g_t, emotion_classifier.label_to_index, ex.d_t)
            s_emotion = emotion_score_head(m_t)
            emo_label = ex.y_emo if ex.y_emo is not None else 0.0
            emo_losses.append(binary_review_loss(s_emotion, torch.tensor(emo_label)))
            emo_mask.append(1.0 if ex.y_emo is not None else 0.0)

        per_sample_losses = {
            "cb": cb_losses,
            "g": torch.stack(g_losses),
            "b": torch.stack(b_losses),
            "e": torch.stack(e_losses),
            "emo": torch.stack(emo_losses),
        }
        masks = {
            "cb": cb_mask,
            "g": torch.ones(len(EXAMPLES)),
            "b": torch.ones(len(EXAMPLES)),
            "e": torch.ones(len(EXAMPLES)),
            "emo": torch.tensor(emo_mask),
        }
        loss = masked_multitask_loss(per_sample_losses, masks)
        loss.backward()
        optimizer.step()
        print(f"  epoch {epoch}: masked multi-task L = {loss.item():.4f}")

    # ---------------------------------------------------------------
    # Stage 3 (Section 12.3): freeze the shared encoder and task heads,
    # then calibrate the Safety Score weights, then freeze those and
    # calibrate the Overall Score weights.
    # ---------------------------------------------------------------
    print("\n=== Stage 3: calibration ===")
    for module in trainable_modules:
        freeze(module)
        module.eval()

    fixed_scores = []
    with torch.no_grad():
        for ex in EXAMPLES:
            h = message_encoder([ex.speaker_id] * len(ex.texts), ex.texts)
            z, _ = conversation_encoder.encode(h)
            safety_tensor = torch.tensor(ex.safety_features)
            h_prev_tensor = torch.tensor(ex.h_prev)

            p_cb = cyberbullying_head.forward_stage1(h)
            risk = cyberbullying_head.risk(p_cb)
            s_cb = cyberbullying_head.window_score(risk)

            s_g, _ = grooming_head(z, safety_tensor)
            s_e = early_detection_head(z, safety_tensor, h_prev_tensor)

            rule_flags = [v >= 0.5 for v in ex.safety_features[6:]]
            s_r_tilde = torch.tensor(rule_safety_score(RuleSignals(*rule_flags)))

            g_i = emotion_classifier(ex.texts)
            g_t = max_mean_top3(g_i.transpose(0, 1))
            m_t = map_emotions(g_t, emotion_classifier.label_to_index, ex.d_t)
            s_emotion = emotion_score_head(m_t)

            fixed_scores.append((s_cb, s_g, s_e, s_r_tilde, s_emotion, ex.y_overall))

    print("-- calibrating Safety Score weights (b_s, v_cb, v_g, v_e, v_r, v_ge) --")
    optimizer_safety = optim.AdamW(risk_fusion.safety_fusion.parameters(), lr=1e-2)
    for epoch in range(20):
        optimizer_safety.zero_grad()
        losses = [
            binary_review_loss(risk_fusion.safety_fusion(s_cb, s_g, s_e, s_r_tilde), torch.tensor(y))
            for s_cb, s_g, s_e, s_r_tilde, _, y in fixed_scores
        ]
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer_safety.step()
        if epoch % 5 == 0 or epoch == 19:
            print(f"  epoch {epoch}: safety calibration loss = {loss.item():.4f}")

    freeze(risk_fusion.safety_fusion)

    print("-- calibrating Overall Score weights (b_o, w_s, w_m) --")
    optimizer_overall = optim.AdamW(risk_fusion.overall_fusion.parameters(), lr=1e-2)
    for epoch in range(20):
        optimizer_overall.zero_grad()
        losses = []
        for s_cb, s_g, s_e, s_r_tilde, s_emotion, y in fixed_scores:
            s_safety = risk_fusion.safety_fusion(s_cb, s_g, s_e, s_r_tilde)
            s_overall = risk_fusion.overall_fusion(s_safety, s_emotion)
            losses.append(binary_review_loss(s_overall, torch.tensor(y)))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer_overall.step()
        if epoch % 5 == 0 or epoch == 19:
            print(f"  epoch {epoch}: overall calibration loss = {loss.item():.4f}")

    # ---------------------------------------------------------------
    # Save the trained weights, then reload them to prove the saved
    # checkpoint reproduces the same prediction.
    # ---------------------------------------------------------------
    output_dir = Path(__file__).parent / "trained_weights_demo"
    output_dir.mkdir(exist_ok=True)
    torch.save(message_encoder.state_dict(), output_dir / "message_encoder.pt")
    torch.save(conversation_encoder.state_dict(), output_dir / "conversation_encoder.pt")
    torch.save(cyberbullying_head.state_dict(), output_dir / "cyberbullying_head.pt")
    torch.save(grooming_head.state_dict(), output_dir / "grooming_head.pt")
    torch.save(early_detection_head.state_dict(), output_dir / "early_detection_head.pt")
    torch.save(emotion_score_head.state_dict(), output_dir / "emotion_score_head.pt")
    torch.save(risk_fusion.state_dict(), output_dir / "risk_fusion.pt")
    print(f"\n=== Saved trained weights to {output_dir} ===")

    print("=== Reloading risk_fusion.pt and re-running one example ===")
    reloaded_fusion = RiskFusion()
    reloaded_fusion.load_state_dict(torch.load(output_dir / "risk_fusion.pt"))
    reloaded_fusion.eval()

    s_cb, s_g, s_e, s_r_tilde, s_emotion, y = fixed_scores[2]  # the grooming example
    with torch.no_grad():
        s_safety = reloaded_fusion.safety_fusion(s_cb, s_g, s_e, s_r_tilde)
        r_t = reloaded_fusion.overall_fusion(s_safety, s_emotion)
    print(f"  example 3 (grooming scenario): Safety={s_safety.item():.3f}  Overall={r_t.item():.3f}  (label={y})")


if __name__ == "__main__":
    main()
