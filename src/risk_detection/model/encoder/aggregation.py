import torch


def max_mean_top3(
    values: torch.Tensor,
    weight_max: float = 0.6,
    weight_top3: float = 0.4,
    top_k: int = 3,
) -> torch.Tensor:
    """0.6 * max_i(values) + 0.4 * MeanTop3(values), over the last dimension.

    Shared by the Cyberbullying Head's window aggregation (Section 5) and,
    per Section 10.2, the Emotion Branch's per-dimension aggregation. Works
    on an unbatched window (T,) -> scalar, or a batch of windows (B, T) -> (B,).
    """
    if values.numel() == 0:
        return values.new_zeros(values.shape[:-1])

    k = min(top_k, values.shape[-1])
    top_values = torch.topk(values, k, dim=-1).values
    return weight_max * values.max(dim=-1).values + weight_top3 * top_values.mean(dim=-1)
