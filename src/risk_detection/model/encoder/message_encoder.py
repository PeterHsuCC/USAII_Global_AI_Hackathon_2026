import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ...conversation import ConversationWindow

DEFAULT_ENCODER_NAME = "bert-base-uncased"


class MessageEncoder(nn.Module):
    """h_i = Encoder([s_i ; m_i]); shared BERT/DistilBERT encoder (d=768 for BERT Base).

    s_i and m_i are concatenated as plain text before tokenization. Pass a
    pre-built tokenizer/encoder (e.g. a tiny locally constructed model) to
    avoid downloading a checkpoint, such as in tests.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ENCODER_NAME,
        tokenizer: PreTrainedTokenizerBase | None = None,
        encoder: PreTrainedModel | None = None,
        max_length: int = 128,
    ):
        super().__init__()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.encoder = encoder or AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.d = self.encoder.config.hidden_size

    @staticmethod
    def _format_input(speaker_id: str, text: str) -> str:
        return f"{speaker_id}: {text}"

    def forward(self, speaker_ids: list[str], texts: list[str]) -> torch.Tensor:
        """Encode a batch of (s_i, m_i) pairs into h_i in R^d via CLS pooling."""
        device = next(self.encoder.parameters()).device
        formatted = [self._format_input(s, m) for s, m in zip(speaker_ids, texts)]
        encoded = self.tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        outputs = self.encoder(**encoded)
        return outputs.last_hidden_state[:, 0, :]

    def encode_window(self, window: ConversationWindow) -> torch.Tensor:
        """Encode every message in a Conversation Window; returns (len(window), d)."""
        if len(window) == 0:
            device = next(self.encoder.parameters()).device
            return torch.zeros((0, self.d), device=device)
        return self.forward([m.speaker_id for m in window], [m.text for m in window])
