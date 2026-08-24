"""Interfaces shared by model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from openrlhf_agent.utils.types import Action


@dataclass(slots=True)
class GenerationResult:
    """Normalized completion output used by training trajectories."""

    text: str
    token_ids: list[int]
    token_logprobs: list[float] | None = None
    finish_reason: str | None = None
    meta_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_logprobs is not None and len(self.token_ids) != len(
            self.token_logprobs
        ):
            raise ValueError(
                "Generated token IDs and log probabilities must have the same length."
            )


class CompletionBackend(ABC):
    """Backend that generates text from a rendered prompt."""

    @abstractmethod
    async def generate(
        self,
        prompt: str | list[int],
        max_tokens: int | None = None,
        *,
        images: Sequence[Any] | None = None,
        sampling_params: Mapping[str, Any] | None = None,
        return_logprobs: bool = False,
        session_id: str | None = None,
    ) -> GenerationResult:
        """Generate text and exact token metadata.

        ``max_tokens=None`` lets the server choose the generation limit. Requested
        logprobs align one-to-one with token IDs. ``session_id`` identifies related
        requests for provider sessions or routing affinity.
        """

    @abstractmethod
    async def tokenize(
        self,
        prompt: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Convert text into token IDs understood by the backend."""


class ActionBackend(ABC):
    """Backend that generates an action from structured messages."""

    @abstractmethod
    async def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> Action:
        """Generate one assistant action from structured messages."""
