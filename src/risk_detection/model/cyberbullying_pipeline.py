from dataclasses import dataclass

import torch

from ..conversation import ConversationWindow
from .conversation_encoder import ConversationEncoder
from .cyberbullying_head import CyberbullyingHead
from .message_encoder import MessageEncoder


@dataclass
class CyberbullyingResult:
    per_message_risk: torch.Tensor  # p_cb,i^risk, shape (T,)
    window_score: torch.Tensor  # S_cb(t), scalar
    attention_weights: torch.Tensor  # alpha_i, shape (T,)


class CyberbullyingPipeline:
    """Ties the Message/Conversation encoders (Section 4) to the
    Cyberbullying Head (Section 5) for one rolling Conversation Window,
    matching steps 3 and 5 of the Stage 2 Integrated Inference Flow."""

    def __init__(
        self,
        message_encoder: MessageEncoder,
        conversation_encoder: ConversationEncoder,
        head: CyberbullyingHead,
        use_context: bool = True,
    ):
        self.message_encoder = message_encoder
        self.conversation_encoder = conversation_encoder
        self.head = head
        self.use_context = use_context

    @torch.no_grad()
    def score(self, window: ConversationWindow) -> CyberbullyingResult:
        h = self.message_encoder.encode_window(window)
        if h.shape[0] == 0:
            empty = h.new_zeros(0)
            return CyberbullyingResult(empty, h.new_zeros(()), empty)

        if self.use_context:
            z, alpha = self.conversation_encoder.encode(h)
            p = self.head.forward_stage2(h, z)
        else:
            alpha = h.new_zeros(h.shape[0])
            p = self.head.forward_stage1(h)

        risk = self.head.risk(p)
        return CyberbullyingResult(risk, self.head.window_score(risk), alpha)
