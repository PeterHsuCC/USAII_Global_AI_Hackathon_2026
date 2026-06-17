"""Builds a tiny, randomly-initialized multi-label sequence classifier
(tokenizer + model) entirely in memory, with a GoEmotions-shaped label set,
so emotion-branch tests don't need network access or a downloaded
checkpoint."""

import tempfile

from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

_VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "i",
    "am",
    "scared",
    "sad",
    "angry",
    "love",
    "you",
    "care",
    "about",
    "this",
]

_LABELS = (
    "admiration",
    "fear",
    "nervousness",
    "grief",
    "sadness",
    "anger",
    "caring",
    "love",
    "neutral",
)


def make_tiny_emotion_classifier(hidden_size: int = 8):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(_VOCAB))
        vocab_path = f.name

    tokenizer = BertTokenizer(vocab_file=vocab_path)
    config = BertConfig(
        vocab_size=len(_VOCAB),
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=16,
        max_position_embeddings=32,
        num_labels=len(_LABELS),
        id2label={i: label for i, label in enumerate(_LABELS)},
        label2id={label: i for i, label in enumerate(_LABELS)},
        problem_type="multi_label_classification",
    )
    model = BertForSequenceClassification(config)
    model.eval()
    return tokenizer, model
