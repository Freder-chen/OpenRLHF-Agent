"""Assistant action parsed from the LLM response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence

from .conversation import Message, ToolCall


@dataclass
class Action:
    """Assistant reply split into text and tool calls."""

    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    refusal: Optional[str] = None
    reasoning_content: Optional[str] = None


@dataclass
class Observation:
    """Outcome produced after applying an action to the environment."""

    step_index: int
    feedback_messages: Optional[List[Message]] = None
    feedback_text: str | None = None
    done: bool = False


@dataclass
class RewardSample:
    """Encapsulates the question, process history, and reference result."""

    question: Optional[Any] = None
    process_messages: Optional[Sequence[Mapping[str, Any]]] = None
    # result: Optional[Any] = None
