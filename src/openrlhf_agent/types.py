"""Compatibility layer that re-exports the core agent models."""

from openrlhf_agent.core import ChatMessage, ParsedAssistantAction, ToolCall

__all__ = ["ToolCall", "ParsedAssistantAction", "ChatMessage"]
