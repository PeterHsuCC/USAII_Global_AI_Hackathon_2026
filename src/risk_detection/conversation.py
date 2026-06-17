from dataclasses import dataclass, field


@dataclass(frozen=True)
class Message:
    """One element of the window tuple (s_i, m_i, Delta t_i)."""

    speaker_id: str
    text: str
    relative_time: float


@dataclass
class ConversationWindow:
    """Rolling Conversation Window X_t = {(s_i, m_i, Delta t_i)}_{i=t-k+1}^{t}.

    Holds at most the k most recent messages; older messages are dropped
    as new ones arrive.
    """

    k: int
    messages: list[Message] = field(default_factory=list)

    def add(self, message: Message) -> None:
        self.messages.append(message)
        if len(self.messages) > self.k:
            self.messages = self.messages[-self.k :]

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)
