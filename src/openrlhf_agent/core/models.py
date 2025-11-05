"""Shared domain models used across the agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """One tool invocation requested by the model."""

    id: str
    name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    refusal: Optional[str] = None


class ParsedAssistantAction(BaseModel):
    """Assistant reply split into text and tool calls."""

    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    refusal: Optional[str] = None


class ChatMessage(BaseModel):
    """Single chat turn tracked inside the session memory."""

    role: str
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    reasoning_content: Optional[str] = None  # used by reasoning-capable backends


@dataclass
class AgentStepResult:
    """Outcome produced after applying an action to the environment."""

    idx: int
    assistant_message: ChatMessage
    tool_messages: List[ChatMessage] = field(default_factory=list)
    reward: float = 0.0
    terminated: bool = False


__all__ = ["ToolCall", "ParsedAssistantAction", "ChatMessage", "AgentStepResult"]

