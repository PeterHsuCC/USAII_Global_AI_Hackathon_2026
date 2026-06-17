import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ..conversation import ConversationWindow

DEFAULT_GOEMOTIONS_MODEL = "SamLowe/roberta-base-go_emotions"
NEUTRAL_LABEL = "neutral"


class GoEmotionsClassifier(nn.Module):
    """G_i = GoEmotionsClassifier(m_i) in [0,1]^d_G, applied per message
    (Section 10.2).

    Wraps a pretrained multi-label GoEmotions checkpoint. d_G and label
    order are read from the checkpoint's published config (id2label) at
    load time -- never hardcoded -- since the report notes d_G is 27 or 28
    depending on whether the checkpoint includes `neutral`. The `neutral`
    label, if present, is dropped from the returned vector since it does
    not contribute to the project's five-dimensional mapping (phi).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GOEMOTIONS_MODEL,
        tokenizer: PreTrainedTokenizerBase | None = None,
        encoder: PreTrainedModel | None = None,
        max_length: int = 128,
    ):
        super().__init__()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.encoder = encoder or AutoModelForSequenceClassification.from_pretrained(model_name)
        self.max_length = max_length

        id2label = self.encoder.config.id2label
        all_labels = [id2label[i] for i in range(len(id2label))]
        neutral_index = next(
            (i for i, label in enumerate(all_labels) if label.lower() == NEUTRAL_LABEL), None
        )
        self._keep_indices = [i for i in range(len(all_labels)) if i != neutral_index]

        self.label_names = [all_labels[i] for i in self._keep_indices]
        self.d_g = len(self.label_names)
        self.label_to_index = {label.lower(): i for i, label in enumerate(self.label_names)}

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of messages into G_i in [0,1]^d_G each (neutral
        dropped if the checkpoint has it)."""
        device = next(self.encoder.parameters()).device
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        logits = self.encoder(**encoded).logits
        probs = torch.sigmoid(logits)  # multi-label: independent sigmoid per label
        return probs[:, self._keep_indices]

    def encode_window(self, window: ConversationWindow) -> torch.Tensor:
        """Encode every message in the window; returns (len(window), d_g)."""
        if len(window) == 0:
            device = next(self.encoder.parameters()).device
            return torch.zeros((0, self.d_g), device=device)
        return self.forward([m.text for m in window])
