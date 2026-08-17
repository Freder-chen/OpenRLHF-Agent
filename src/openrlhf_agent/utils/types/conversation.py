"""Conversation helper plus supporting message and tool-call models."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from pydantic import BaseModel


class ToolCall(BaseModel):
    """One tool invocation requested by the model."""

    call_id: str
    name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    refusal: Optional[str] = None


class Message(BaseModel):
    """Single chat turn tracked inside the session memory."""

    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    reasoning_content: Optional[str] = None  # used by reasoning-capable backends


class Conversation:
    """Stores chat messages and knows how to render them."""

    def __init__(self) -> None:
        self._messages: List[Message] = []

    def reset(self, *, system_prompt: str) -> None:
        """Start a fresh transcript using the provided system prompt."""

        self._messages = [Message(role="system", content=system_prompt)]

    def extend(self, messages: Iterable[Message | Mapping[str, Any]]) -> None:
        """Append a list of historical messages."""

        for message in messages:
            if isinstance(message, Message):
                self._messages.append(message)
            elif isinstance(message, Mapping):
                self._messages.append(Message(**message))

    def append(self, message: Message) -> None:
        """Append a message and return it for convenience."""

        self._messages.append(message)

    @property
    def messages(self) -> List[dict]:
        """Expose a shallow copy for inspection or debugging."""

        return [message.model_dump(exclude_none=True) for message in self._messages]
