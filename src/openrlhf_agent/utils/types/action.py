"""Assistant action parsed from the LLM response."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .conversation import Message, ToolCall


@dataclass
class Action:
    """Assistant reply split into text and tool calls."""

    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    reasoning_content: Optional[str] = None
    error: Optional[str] = None

    def to_message(self) -> Message:
        """Convert the action into an assistant message."""

        return Message(
            role="assistant",
            content=self.content or None,
            tool_calls=self.tool_calls or None,
            reasoning_content=self.reasoning_content or None,
        )


@dataclass
class Observation:
    """Outcome produced after applying an action to the environment."""

    step_index: int
    feedback_messages: List[Message] = field(default_factory=list)
    feedback_text: str = ""
    done: bool = False
    environment_images: list[Any] = field(default_factory=list)
