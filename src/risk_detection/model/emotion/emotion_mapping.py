import torch

DISTRESS_WEIGHTS = {"fear": 0.30, "nervousness": 0.30, "grief": 0.20, "sadness": 0.20}
DEPENDENCY_PROXY_WEIGHTS = {"caring": 0.5, "love": 0.5}
DEFAULT_LAMBDA = 0.1

MAPPED_EMOTION_NAMES = ("fear", "sadness", "anger", "distress", "dependency")


def map_emotions(
    g_t: torch.Tensor,
    label_to_index: dict[str, int],
    d_t: torch.Tensor | float,
    lam: float = DEFAULT_LAMBDA,
) -> torch.Tensor:
    """M_t = phi(G_t, D_t) = [Fear_t, Sadness_t, Anger_t, Distress_t,
    Dependency_t] in [0,1]^5 (Section 10.2).

    g_t: (..., d_G) aggregated GoEmotions vector for the window (G_t).
    label_to_index: maps a (lowercase) GoEmotions label name to its column
        in g_t -- e.g. GoEmotionsClassifier.label_to_index.
    d_t: D_t, the separate LLM-extracted emotional-dependency signal.
    lam: weight on the minimal GoEmotions-based dependency proxy vs. D_t;
        prototype default 0.1 so D_t dominates this dimension.
    """

    def col(name: str) -> torch.Tensor:
        return g_t[..., label_to_index[name]]

    fear = col("fear")
    sadness = col("sadness")
    anger = col("anger")

    distress = (
        DISTRESS_WEIGHTS["fear"] * col("fear")
        + DISTRESS_WEIGHTS["nervousness"] * col("nervousness")
        + DISTRESS_WEIGHTS["grief"] * col("grief")
        + DISTRESS_WEIGHTS["sadness"] * col("sadness")
    )

    dependency_proxy = (
        DEPENDENCY_PROXY_WEIGHTS["caring"] * col("caring")
        + DEPENDENCY_PROXY_WEIGHTS["love"] * col("love")
    )
    d_t = torch.as_tensor(d_t, dtype=g_t.dtype, device=g_t.device)
    dependency = lam * dependency_proxy + (1.0 - lam) * d_t

    return torch.stack([fear, sadness, anger, distress, dependency], dim=-1)
