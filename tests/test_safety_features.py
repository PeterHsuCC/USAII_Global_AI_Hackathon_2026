from risk_detection import (
    ConversationWindow,
    LLMSafetySignals,
    Message,
    RuleSignalExtractor,
    SafetyFeatureExtractor,
)


class _StubLLMExtractor:
    def extract(self, window: ConversationWindow) -> LLMSafetySignals:
        return LLMSafetySignals(
            secrecy=0.9,
            isolation=0.1,
            dependency=0.2,
            sexual_escalation=0.0,
            threat=0.0,
            coercion=0.3,
        )


def test_combined_vector_has_eleven_dimensions():
    window = ConversationWindow(k=2)
    window.add(Message(speaker_id="a", text="our little secret ok?", relative_time=0.0))
    window.add(Message(speaker_id="b", text="ok i promise", relative_time=1.0))

    extractor = SafetyFeatureExtractor(
        llm_extractor=_StubLLMExtractor(),
        rule_extractor=RuleSignalExtractor(),
    )
    features = extractor.extract(window)
    vector = features.to_vector()

    assert len(vector) == 11
    assert vector[:6] == [0.9, 0.1, 0.2, 0.0, 0.0, 0.3]
    assert vector[6] == 1.0  # secret_request triggered by "our little secret"


def test_llm_signal_vector_is_clamped_to_unit_interval():
    signals = LLMSafetySignals(
        secrecy=1.5,
        isolation=-0.2,
        dependency=0.5,
        sexual_escalation=0.0,
        threat=0.0,
        coercion=0.0,
    )

    assert signals.to_vector() == [1.0, 0.0, 0.5, 0.0, 0.0, 0.0]
