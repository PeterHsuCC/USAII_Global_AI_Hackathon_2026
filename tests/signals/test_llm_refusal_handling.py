import pytest

from risk_detection import ConversationWindow, LLMRefusalError, LLMSafetySignals, Message
from risk_detection.signals.emotional_dependency import EmotionalDependencyExtractor, EmotionalDependencySignal
from risk_detection.signals.llm_safety import LLMSafetySignalExtractor


class _FakeStopDetails:
    def __init__(self, category):
        self.category = category


class _FakeResponse:
    def __init__(self, stop_reason, parsed_output=None, category=None):
        self.stop_reason = stop_reason
        self.parsed_output = parsed_output
        self.stop_details = _FakeStopDetails(category) if stop_reason == "refusal" else None


class _FakeMessages:
    def __init__(self, response):
        self._response = response

    def parse(self, **kwargs):
        return self._response


class _FakeClient:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


def _window() -> ConversationWindow:
    window = ConversationWindow(k=2)
    window.add(Message(speaker_id="a", text="hello", relative_time=0.0))
    return window


def test_llm_safety_extractor_raises_refusal_error_with_category():
    client = _FakeClient(_FakeResponse("refusal", category="bio"))
    extractor = LLMSafetySignalExtractor(client=client)

    with pytest.raises(LLMRefusalError) as exc_info:
        extractor.extract(_window())

    assert exc_info.value.category == "bio"


def test_llm_safety_extractor_raises_runtime_error_on_none_output_without_refusal():
    client = _FakeClient(_FakeResponse("max_tokens", parsed_output=None))
    extractor = LLMSafetySignalExtractor(client=client)

    with pytest.raises(RuntimeError):
        extractor.extract(_window())


def test_llm_safety_extractor_returns_parsed_output_on_success():
    signals = LLMSafetySignals(
        secrecy=0.1, isolation=0.0, dependency=0.0, sexual_escalation=0.0, threat=0.0, coercion=0.0
    )
    client = _FakeClient(_FakeResponse("end_turn", parsed_output=signals))
    extractor = LLMSafetySignalExtractor(client=client)

    assert extractor.extract(_window()) is signals


def test_emotional_dependency_extractor_raises_refusal_error_with_category():
    client = _FakeClient(_FakeResponse("refusal", category="sexual_content"))
    extractor = EmotionalDependencyExtractor(client=client)

    with pytest.raises(LLMRefusalError) as exc_info:
        extractor.extract(_window())

    assert exc_info.value.category == "sexual_content"


def test_emotional_dependency_extractor_raises_runtime_error_on_none_output_without_refusal():
    client = _FakeClient(_FakeResponse("max_tokens", parsed_output=None))
    extractor = EmotionalDependencyExtractor(client=client)

    with pytest.raises(RuntimeError):
        extractor.extract(_window())


def test_emotional_dependency_extractor_returns_value_on_success():
    client = _FakeClient(_FakeResponse("end_turn", parsed_output=EmotionalDependencySignal(dependency=0.4)))
    extractor = EmotionalDependencyExtractor(client=client)

    assert extractor.extract(_window()) == 0.4
