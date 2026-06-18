import torch

from risk_detection.model import (
    OverallScoreFusion,
    PrototypeRiskFusion,
    PrototypeSafetyScoreFusion,
    RiskFusion,
    SafetyScoreFusion,
)


def test_safety_score_fusion_shape_and_range():
    fusion = SafetyScoreFusion()
    s_cb = torch.rand(4)
    s_g = torch.rand(4)
    s_e = torch.rand(4)
    s_r_tilde = torch.rand(4)

    s_safety = fusion(s_cb, s_g, s_e, s_r_tilde)

    assert s_safety.shape == (4,)
    assert (s_safety >= 0).all() and (s_safety <= 1).all()


def test_safety_score_fusion_matches_hand_calculation_with_known_weights():
    fusion = SafetyScoreFusion()
    with torch.no_grad():
        # order: [S_cb, S_g, S_e, S~_r, S_g*S_e]
        fusion.linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))
        fusion.linear.bias.copy_(torch.tensor([0.5]))

    s_cb, s_g, s_e, s_r_tilde = (
        torch.tensor(0.1),
        torch.tensor(0.2),
        torch.tensor(0.3),
        torch.tensor(0.4),
    )
    s_safety = fusion(s_cb, s_g, s_e, s_r_tilde)

    s_ge = 0.2 * 0.3
    expected_logit = 0.5 + 1.0 * 0.1 + 2.0 * 0.2 + 3.0 * 0.3 + 4.0 * 0.4 + 5.0 * s_ge
    expected = torch.sigmoid(torch.tensor(expected_logit))
    assert torch.allclose(s_safety, expected, atol=1e-6)


def test_overall_score_fusion_shape_and_range():
    fusion = OverallScoreFusion()
    s_safety = torch.rand(4)
    s_emotion = torch.rand(4)

    r_t = fusion(s_safety, s_emotion)

    assert r_t.shape == (4,)
    assert (r_t >= 0).all() and (r_t <= 1).all()


def test_risk_fusion_returns_three_dashboard_scores():
    fusion = RiskFusion()
    s_cb = torch.tensor(0.7)
    s_g = torch.tensor(0.8)
    s_e = torch.tensor(0.6)
    s_r_tilde = torch.tensor(0.5)
    s_emotion = torch.tensor(0.4)

    fused = fusion(s_cb, s_g, s_e, s_r_tilde, s_emotion)

    assert fused.emotion_score is s_emotion
    assert 0.0 <= fused.safety_score.item() <= 1.0
    assert 0.0 <= fused.overall_score.item() <= 1.0

    # Overall score should equal sigmoid(b_o + w_s*S_safety + w_m*S_emotion)
    # computed from the *same* safety score this call produced.
    expected_overall = fusion.overall_fusion(fused.safety_score, fused.emotion_score)
    assert torch.allclose(fused.overall_score, expected_overall)


def test_safety_and_overall_fusion_have_independent_weights():
    fusion = RiskFusion()

    assert fusion.safety_fusion.linear is not fusion.overall_fusion.linear
    safety_params = set(map(id, fusion.safety_fusion.parameters()))
    overall_params = set(map(id, fusion.overall_fusion.parameters()))
    assert safety_params.isdisjoint(overall_params)


def test_prototype_safety_score_fusion_does_not_take_s_e():
    fusion = PrototypeSafetyScoreFusion()
    s_cb = torch.rand(4)
    s_g = torch.rand(4)
    s_r_tilde = torch.rand(4)

    s_safety = fusion(s_cb, s_g, s_r_tilde)

    assert s_safety.shape == (4,)
    assert (s_safety >= 0).all() and (s_safety <= 1).all()
    assert fusion.linear.in_features == 3


def test_prototype_safety_score_fusion_matches_hand_calculation():
    fusion = PrototypeSafetyScoreFusion()
    with torch.no_grad():
        # order: [S_cb, S_g, S~_r]
        fusion.linear.weight.copy_(torch.tensor([[1.0, 2.0, 4.0]]))
        fusion.linear.bias.copy_(torch.tensor([0.5]))

    s_cb, s_g, s_r_tilde = torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.4)
    s_safety = fusion(s_cb, s_g, s_r_tilde)

    expected_logit = 0.5 + 1.0 * 0.1 + 2.0 * 0.2 + 4.0 * 0.4
    expected = torch.sigmoid(torch.tensor(expected_logit))
    assert torch.allclose(s_safety, expected, atol=1e-6)


def test_prototype_risk_fusion_returns_three_dashboard_scores_without_s_e():
    fusion = PrototypeRiskFusion()
    s_cb = torch.tensor(0.7)
    s_g = torch.tensor(0.8)
    s_r_tilde = torch.tensor(0.5)
    s_emotion = torch.tensor(0.4)

    fused = fusion(s_cb, s_g, s_r_tilde, s_emotion)

    assert fused.emotion_score is s_emotion
    assert 0.0 <= fused.safety_score.item() <= 1.0
    assert 0.0 <= fused.overall_score.item() <= 1.0
    expected_overall = fusion.overall_fusion(fused.safety_score, fused.emotion_score)
    assert torch.allclose(fused.overall_score, expected_overall)


def test_prototype_and_full_safety_fusion_have_independent_weights():
    full = RiskFusion()
    prototype = PrototypeRiskFusion()

    full_params = set(map(id, full.safety_fusion.parameters()))
    prototype_params = set(map(id, prototype.safety_fusion.parameters()))
    assert full_params.isdisjoint(prototype_params)
