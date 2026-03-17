"""Core interfaces shared by engine backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union


class LLMEngine(ABC):
    """Defines minimal methods required for a language model backend."""

    @abstractmethod
    async def generate(
        self,
        prompt: Optional[Union[str, List[int]]],
        max_tokens: int = 10240,
        temperature: float = 0.6,
        stream: bool = False,
    ) -> Tuple[List[int], str]:
        """Return generated token ids and the decoded text."""

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[Dict[str, str]],
        max_tokens: int = 10240,
        temperature: float = 0.6,
        stream: bool = False,
    ) -> str:
        """Return decoded text from a chat-style completion call."""

    @abstractmethod
    async def tokenize(self, prompt: str) -> List[int]:
        """Convert text into token ids understood by the backend."""


__all__ = ["LLMEngine"]
