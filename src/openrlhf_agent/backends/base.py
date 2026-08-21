"""Core interfaces shared by language model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from openrlhf_agent.utils.types import Action

if TYPE_CHECKING:
    from openrlhf_agent.backends.openai.vllm.protocols import Protocol


class CompletionBackend(ABC):
    """Backend that generates text from a rendered prompt."""

    protocol: "Protocol"

    @abstractmethod
    async def generate(
        self,
        prompt: str | list[int],
        max_tokens: int | None = None,
    ) -> tuple[list[int], str]:
        """Return generated token ids and the decoded text."""

    @abstractmethod
    async def tokenize(self, prompt: str) -> list[int]:
        """Convert text into token ids understood by the backend."""


class ChatBackend(ABC):
    """Backend that generates an action from structured chat messages."""

    @abstractmethod
    async def generate_chat(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Action:
        """Generate one structured assistant action from chat messages."""
