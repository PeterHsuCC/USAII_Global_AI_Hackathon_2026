import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ConversationEncoder(nn.Module):
    """u_i = BiGRU(h_i, u_{i-1}); e_i = v^T tanh(W_a u_i + b_a); alpha = softmax(e);
    z_t = sum_i alpha_i u_i.

    hidden_size is d // 2 so the bidirectional GRU's forward+backward
    concatenation has dimension d, matching the prototype setting d_z = d
    (Section 4.2).
    """

    def __init__(self, d: int = 768):
        super().__init__()
        if d % 2 != 0:
            raise ValueError("d must be even so the BiGRU output matches d_z = d")
        self.d = d
        self.gru = nn.GRU(
            input_size=d,
            hidden_size=d // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.W_a = nn.Linear(d, d)
        self.v = nn.Linear(d, 1, bias=False)

    def forward(
        self, h: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """h: (B, T, d) message encodings for a batch of conversation windows.

        Returns (z, alpha): z is (B, d), the conversation representation;
        alpha is (B, T), the attention weights (0 at padded positions).
        """
        batch_size, max_len, _ = h.shape
        if max_len == 0:
            return h.new_zeros((batch_size, self.d)), h.new_zeros((batch_size, 0))

        if lengths is None:
            lengths = torch.full((batch_size,), max_len, dtype=torch.long)
        lengths = lengths.to(torch.long)

        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        u, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=max_len)

        e = self.v(torch.tanh(self.W_a(u))).squeeze(-1)  # (B, T)
        positions = torch.arange(max_len, device=h.device).unsqueeze(0)
        mask = positions < lengths.to(h.device).unsqueeze(1)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=1)

        z = torch.einsum("bt,btd->bd", alpha, u)
        return z, alpha

    def encode(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience wrapper for a single, unbatched conversation window.

        h: (T, d) -> (z: (d,), alpha: (T,)).
        """
        z, alpha = self.forward(h.unsqueeze(0))
        return z.squeeze(0), alpha.squeeze(0)
