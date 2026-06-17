"""Builds a rolling Conversation Window and runs Section 3 signal extraction.

Requires ANTHROPIC_API_KEY to be set in the environment (used by the LLM
safety signal extractor).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from risk_detection import ConversationWindow, Message, SafetyFeatureExtractor

CONVERSATION = [
    ("user_a", "hey, you seem really stressed lately"),
    ("user_b", "yeah school has been a lot"),
    ("user_a", "you can talk to me about anything, i'm the only one who gets you"),
    ("user_a", "let's keep this between us though, ok? don't tell your parents"),
    ("user_a", "add me on snapchat so we can talk more privately"),
]


def main() -> None:
    window = ConversationWindow(k=10)
    for i, (speaker, text) in enumerate(CONVERSATION):
        window.add(Message(speaker_id=speaker, text=text, relative_time=float(i)))

    extractor = SafetyFeatureExtractor()
    features = extractor.extract(window)

    print("LLM safety signals (L_t):", features.llm_signals.to_vector())
    print("Rule signals (Q_t):      ", features.rule_signals.to_vector())
    print("Combined F_t^safe (R^11):", features.to_vector())


if __name__ == "__main__":
    main()
