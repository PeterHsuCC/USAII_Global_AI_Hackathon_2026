from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

DEFAULT_MC_DROPOUT_PASSES = 20
DEFAULT_R_THRESHOLD = 0.7
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_RULE_THRESHOLD = 0.8


def enable_mc_dropout(module: nn.Module) -> None:
    """Put `module` in eval mode except for its nn.Dropout submodules,
    which are switched back to train mode so they stay stochastic at
    inference time -- the standard MC Dropout technique (Section 13).

    Pretrained encoders already contain nn.Dropout layers by default (e.g.
    the BERT backbone behind MessageEncoder, or the RoBERTa backbone
    behind GoEmotionsClassifier), so no extra dropout needs to be added to
    this project's own heads for MC Dropout to produce genuinely
    stochastic forward passes.
    """
    module.eval()
    for submodule in module.modules():
        if isinstance(submodule, nn.Dropout):
            submodule.train()


@dataclass
class UncertaintyEstimate:
    mean: torch.Tensor  # R_hat_t
    variance: torch.Tensor  # v_t
    uncertainty: torch.Tensor  # U_t
    confidence: torch.Tensor  # confidence_t


def mc_dropout_stats(
    predict_fn: Callable[[], torch.Tensor],
    n: int = DEFAULT_MC_DROPOUT_PASSES,
) -> UncertaintyEstimate:
    """Run N stochastic forward passes via `predict_fn` and compute
    R_hat_t, v_t, U_t, confidence_t (Section 13).

    predict_fn must perform one stochastic forward pass (e.g. with MC
    Dropout enabled via enable_mc_dropout on the relevant module(s)) and
    return R_t^(n) as a tensor, scalar or batched. Because every
    prediction lies in [0,1], the population variance is at most 0.25;
    U_t = min(1, 4*v_t) maps that range onto [0,1].

    This estimate primarily captures uncertainty from the trainable
    safety and fusion components actually exercised by `predict_fn`. It
    does not represent uncertainty in a frozen, non-dropout component
    (e.g. a frozen GoEmotions classifier with dropout disabled) unless
    that component is also put into MC Dropout mode.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    with torch.no_grad():
        samples = torch.stack([predict_fn() for _ in range(n)], dim=0)  # (n, ...)

    r_hat = samples.mean(dim=0)
    variance = ((samples - r_hat) ** 2).mean(dim=0)
    uncertainty = torch.clamp(4.0 * variance, max=1.0)
    confidence = 1.0 - uncertainty
    return UncertaintyEstimate(mean=r_hat, variance=variance, uncertainty=uncertainty, confidence=confidence)


def human_review_required(
    r_hat: torch.Tensor | float,
    confidence: torch.Tensor | float,
    s_r_tilde: torch.Tensor | float,
    r_threshold: float = DEFAULT_R_THRESHOLD,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    rule_threshold: float = DEFAULT_RULE_THRESHOLD,
) -> torch.Tensor:
    """Review_t = 1(R_hat_t > 0.7) v 1(confidence_t < 0.6) v
    1(S~_r(t) >= 0.8) (Section 13).

    Thresholds are prototype settings; tune with validation data and
    operational review capacity. Returns a boolean tensor (call .item()
    for a single window).
    """
    r_hat = torch.as_tensor(r_hat)
    confidence = torch.as_tensor(confidence)
    s_r_tilde = torch.as_tensor(s_r_tilde)
    return (r_hat > r_threshold) | (confidence < confidence_threshold) | (s_r_tilde >= rule_threshold)
