"""Assistant action parsed from the LLM response."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .conversation import Message, ToolCall


@dataclass
class Action:
    """Assistant reply split into text and tool calls."""

    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    refusal: Optional[str] = None
    reasoning_content: Optional[str] = None


class Status(Enum):
    CONTINUE = "continue"
    DONE = "done"


@dataclass
class Observation:
    """Outcome produced after applying an action to the environment."""

    feedback_messages: Optional[List[Message]] = None
    feedback_text: str | None = None
    status: Status = Status.CONTINUE
