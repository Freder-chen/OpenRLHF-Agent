"""Shared Pydantic models used by the agent runtime."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """One tool call that can be sent to the tool runner."""
    id: str
    name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    refusal: Optional[str] = None


class ParsedAssistantAction(BaseModel):
    """Assistant reply split into plain text and tool calls."""
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    refusal: Optional[str] = None


class ChatMessage(BaseModel):
    """Single chat turn stored by the conversation memory."""
    role: str
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    reasoning_content: Optional[str] = None  # non-OpenAI field used by vLLM reasoning mode


__all__ = ["ToolCall", "ParsedAssistantAction", "ChatMessage"]
