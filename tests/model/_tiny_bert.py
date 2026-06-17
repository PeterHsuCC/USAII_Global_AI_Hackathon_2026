"""Builds a tiny, randomly-initialized BERT (tokenizer + model) entirely in
memory so tests don't need network access or a downloaded checkpoint."""

import tempfile

from transformers import BertConfig, BertModel, BertTokenizer

_VOCAB = [
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
    "stupid",
    "stop",
    "bullying",
    "me",
    "secret",
    "test",
    "a",
    "b",
    ":",
]


def make_tiny_bert(hidden_size: int = 8):
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
    )
    model = BertModel(config)
    model.eval()
    return tokenizer, model
