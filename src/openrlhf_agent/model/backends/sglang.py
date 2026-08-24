"""SGLang native ``/generate`` backend.

The native endpoint is used instead of an OpenAI-compatible endpoint because
RL rollouts need the exact token IDs and log probabilities produced by the
engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any

import httpx

from openrlhf_agent.model.backends.base import CompletionBackend, GenerationResult


class SGLangCompletionBackend(CompletionBackend):
    """Generate one rollout at a time through SGLang's native HTTP API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 600.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/").removesuffix("/v1"),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
            timeout=timeout,
            transport=transport,
        )

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
        """Generate with exact token IDs and optional log probabilities."""

        payload: dict[str, Any] = {
            "text" if isinstance(prompt, str) else "input_ids": prompt,
            "sampling_params": {
                **(sampling_params or {}),
                "n": 1,
                "max_new_tokens": max_tokens,
            },
            "return_logprob": return_logprobs,
        }
        if images:
            payload["image_data"] = images

        response = await self.client.post(
            "/generate",
            json=payload,
            headers={"X-SMG-Routing-Key": session_id} if session_id else None,
        )
        response.raise_for_status()
        data = response.json()
        meta_info = data["meta_info"]
        records = meta_info["output_token_logprobs"] if return_logprobs else None
        return GenerationResult(
            text=data["text"],
            token_ids=[record[1] for record in records]
            if records is not None
            else data["output_ids"],
            token_logprobs=[record[0] for record in records]
            if records is not None
            else None,
            finish_reason=(meta_info.get("finish_reason") or {}).get("type"),
            meta_info=meta_info,
        )

    async def tokenize(
        self,
        prompt: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Tokenize text with SGLang's OpenAI-compatible tokenizer endpoint."""

        response = await self.client.post(
            "/v1/tokenize",
            json={
                "prompt": prompt,
                "add_special_tokens": add_special_tokens,
            },
        )
        response.raise_for_status()
        return response.json()["tokens"]

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""

        await self.client.aclose()

    async def __aenter__(self) -> SGLangCompletionBackend:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
