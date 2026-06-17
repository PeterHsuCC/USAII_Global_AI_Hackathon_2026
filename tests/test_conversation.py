from risk_detection import ConversationWindow, Message


def test_window_accumulates_up_to_k():
    window = ConversationWindow(k=3)
    for i in range(2):
        window.add(Message(speaker_id="a", text=f"msg{i}", relative_time=float(i)))

    assert len(window) == 2
    assert [m.text for m in window] == ["msg0", "msg1"]


def test_window_drops_oldest_beyond_k():
    window = ConversationWindow(k=3)
    for i in range(5):
        window.add(Message(speaker_id="a", text=f"msg{i}", relative_time=float(i)))

    assert len(window) == 3
    assert [m.text for m in window] == ["msg2", "msg3", "msg4"]
