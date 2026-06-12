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
    reasoning_content: Optional[str] = None


class Conversation:
    """Stores chat messages with a clear boundary between prompt and process."""

    def __init__(self, *, system_prompt: str, messages: Iterable[Message | Mapping[str, Any]] = ()) -> None:
        self._messages: List[Message] = [Message(role="system", content=system_prompt)]
        for item in messages:
            if isinstance(item, Message):
                self._messages.append(item)
            elif isinstance(item, Mapping):
                self._messages.append(Message(**item))
        self.prompt_length: int = len(self._messages)

    def append(self, message: Message) -> None:
        self._messages.append(message)

    def extend(self, messages: Iterable[Message]) -> None:
        self._messages.extend(messages)

    @property
    def messages(self) -> List[dict]:
        return [m.model_dump(exclude_none=True) for m in self._messages]

    @property
    def prompt_messages(self) -> List[dict]:
        return [m.model_dump(exclude_none=True) for m in self._messages[:self.prompt_length]]

    @property
    def process_messages(self) -> List[dict]:
        return [m.model_dump(exclude_none=True) for m in self._messages[self.prompt_length:]]
