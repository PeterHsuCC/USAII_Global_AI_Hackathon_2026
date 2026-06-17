from _tiny_emotion_classifier import make_tiny_emotion_classifier

from risk_detection import ConversationWindow, Message
from risk_detection.model import EmotionPipeline, EmotionScoreHead, GoEmotionsClassifier


class _StubDependencyExtractor:
    def extract(self, window):
        return 0.6


def _make_pipeline() -> EmotionPipeline:
    tokenizer, model = make_tiny_emotion_classifier(hidden_size=8)
    classifier = GoEmotionsClassifier(tokenizer=tokenizer, encoder=model)
    head = EmotionScoreHead()
    return EmotionPipeline(classifier, head, dependency_extractor=_StubDependencyExtractor())


def test_score_returns_scalar_and_mapped_vector():
    pipeline = _make_pipeline()
    window = ConversationWindow(k=5)
    window.add(Message(speaker_id="a", text="i am scared", relative_time=0.0))
    window.add(Message(speaker_id="b", text="i love you", relative_time=1.0))

    result = pipeline.score(window)

    d_g = pipeline.emotion_classifier.d_g
    assert result.emotion_score.shape == ()
    assert result.mapped_emotions.shape == (5,)
    assert result.window_emotions.shape == (d_g,)
    assert result.per_message_emotions.shape == (2, d_g)
    assert result.dependency_signal == 0.6
    assert 0.0 <= result.emotion_score.item() <= 1.0


def test_score_on_empty_window_short_circuits():
    pipeline = _make_pipeline()
    window = ConversationWindow(k=5)

    result = pipeline.score(window)

    assert result.emotion_score.item() == 0.0
    assert result.mapped_emotions.shape == (5,)
    assert result.dependency_signal == 0.0
