"""Interfaces shared by model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from openrlhf_agent.model.protocols.base import CompletionProtocol, RenderedPrompt
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
    """Backend that renders, generates, and parses completion-model output."""

    def __init__(self, *, protocol: CompletionProtocol) -> None:
        self.protocol = protocol

    def render_prompt(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> RenderedPrompt:
        """Render the initial prompt for this backend's model."""

        return self.protocol.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
        )

    def render_feedback(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        environment_messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> RenderedPrompt:
        """Render the incremental prompt after an environment transition."""

        return self.protocol.render_feedback(
            messages=messages,
            environment_messages=environment_messages,
            tools=tools,
        )

    def parse_action(self, text: str) -> Action:
        """Parse generated text according to this backend's model protocol."""

        return self.protocol.parse_action(text)

    @abstractmethod
    async def generate(
        self,
        token_ids: list[int],
        max_tokens: int | None = None,
        *,
        images: Sequence[Any] | None = None,
        sampling_params: Mapping[str, Any] | None = None,
        return_logprobs: bool = False,
        session_id: str | None = None,
    ) -> GenerationResult:
        """Generate text and exact token metadata from token IDs.

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
