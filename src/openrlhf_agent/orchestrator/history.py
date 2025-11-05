"""Lightweight helper to keep the session chat history tidy."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from openrlhf_agent.core import ChatMessage
from openrlhf_agent.template import Template


class ChatHistory:
    """Stores messages and builds prompts for the agent session."""

    def __init__(self) -> None:
        self._messages: List[ChatMessage] = []

    # --------------------------------------------------------------------- setup

    def reset(self, *, system_prompt: str) -> None:
        """Start a new history using the latest system prompt."""

        self._messages = [ChatMessage(role="system", content=system_prompt)]

    def extend(self, messages: Iterable[ChatMessage]) -> None:
        """Append a batch of messages in their current order."""

        for message in messages:
            self._messages.append(message)

    # ------------------------------------------------------------------- mutators

    def add(self, message: ChatMessage) -> None:
        """Append a single message to the history."""

        self._messages.append(message)

    def add_tool_message(self, content: str) -> ChatMessage:
        """Create and append a tool response message."""

        tool_message = ChatMessage(role="tool", content=content)
        self._messages.append(tool_message)
        return tool_message

    # -------------------------------------------------------------------- exports

    def render_prompt(
        self,
        template: Template,
        *,
        tools_manifest: Optional[Sequence[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """Render the history into a provider-specific prompt."""

        message_payload = [
            message.model_dump(exclude_none=True) for message in self._messages
        ]
        return template.render_messages(
            messages=message_payload,
            tools_manifest=tools_manifest,
            add_generation_prompt=add_generation_prompt,
        )

    @property
    def messages(self) -> List[ChatMessage]:
        """Return a shallow copy of the recorded messages."""

        return list(self._messages)


__all__ = ["ChatHistory"]
