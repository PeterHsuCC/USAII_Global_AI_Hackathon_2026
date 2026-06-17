import torch
import torch.nn.functional as F

TASK_NAMES = ("cb", "g", "b", "e", "emo")


def cyberbullying_loss(p: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L_cb = -sum_c y_c log(p_c), per sample (Section 12.1).

    p: (N, num_classes) class probabilities (already softmax-ed, e.g. from
    CyberbullyingHead.forward_stage1/forward_stage2). y: (N,) integer class
    indices (the one-hot label implied by y_c). Returns (N,).
    """
    log_p = torch.log(p.clamp_min(eps))
    return F.nll_loss(log_p, y, reduction="none")


def binary_review_loss(p: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """-[y log(p) + (1-y) log(1-p)], per sample.

    Shared form of L_g, L_e, and L_emo (Section 12.1): each is this same
    binary cross-entropy applied to a different scalar score (S_g, S_e, or
    S_emotion) against its own binary label. p, y: (...,). Returns (...,).
    """
    p = p.clamp(eps, 1.0 - eps)
    y = y.to(p.dtype)
    return -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))


def behavior_loss(b: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L_b = -sum_j [y_j log(B_j) + (1-y_j) log(1-B_j)], per sample
    (Section 12.1).

    b, y: (N, 6) -- the six behavior dimensions. A behavior mask should
    gate this loss at the batch level (Section 12.2) since it only applies
    to genuinely behavior-annotated samples. Returns (N,).
    """
    b = b.clamp(eps, 1.0 - eps)
    y = y.to(b.dtype)
    per_dim = -(y * torch.log(b) + (1.0 - y) * torch.log(1.0 - b))
    return per_dim.sum(dim=-1)


def masked_multitask_loss(
    per_sample_losses: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    task_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """L = sum_q lambda_q * [sum_x m_q^x L_q^x / max(1, sum_x m_q^x)]
    (Section 12.2).

    per_sample_losses[q]: (N,) per-sample loss for task q, e.g. from
    cyberbullying_loss / binary_review_loss / behavior_loss. masks[q]: (N,)
    in {0,1}, 1 where sample x has a label for task q -- a sample
    contributes to a task's loss only if it has that task's label.
    task_weights: lambda_q per task; defaults to 1.0 for every task
    present in per_sample_losses.
    """
    if task_weights is None:
        task_weights = {q: 1.0 for q in per_sample_losses}

    total = None
    for task, loss_q in per_sample_losses.items():
        mask_q = masks[task].to(loss_q.dtype)
        denom = mask_q.sum().clamp_min(1.0)
        masked_mean = (mask_q * loss_q).sum() / denom
        term = task_weights[task] * masked_mean
        total = term if total is None else total + term
    return total
