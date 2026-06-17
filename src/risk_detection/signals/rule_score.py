from .rules import RuleSignals

RULE_SIGNAL_NAMES = (
    "secret_request",
    "contact_migration",
    "age_reference",
    "image_request",
    "threat_phrase",
)


def rule_safety_score(
    rule_signals: RuleSignals,
    weights: list[float] | tuple[float, ...] | None = None,
) -> float:
    """S_r(t) = sum_j rho_j * Q_t,j; S~_r(t) = min(1, S_r(t) / sum_j rho_j)
    (Section 10.1).

    Prototype default: rho_j = 1 for all j (uniform weights), treating all
    rule signals equally. Pass `weights`, ordered as RULE_SIGNAL_NAMES, to
    assign higher weight to more severe signals (e.g. image_request,
    threat_phrase) in deployment. Normalizing by sum(weights) keeps
    S~_r(t) in [0,1] regardless of the chosen weights.
    """
    values = rule_signals.to_vector()
    if weights is None:
        weights = [1.0] * len(values)
    if len(weights) != len(values):
        raise ValueError(f"weights must have length {len(values)}, got {len(weights)}")

    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("sum of weights must be nonzero to normalize S_r(t)")

    s_r = sum(w * q for w, q in zip(weights, values))
    return min(1.0, s_r / total_weight)
