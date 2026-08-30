"""vLLM backend using its token-in/token-out API."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Mapping, Sequence

import httpx

from openrlhf_agent.model.backends.base import CompletionBackend, GenerationResult


class VLLMCompletionBackend(CompletionBackend):
    """Generate text and token IDs through vLLM-specific endpoints."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
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
        payload: dict[str, Any] = {
            "model": self.model,
            "token_ids": await self.tokenize(prompt)
            if isinstance(prompt, str)
            else prompt,
            "sampling_params": {
                **(sampling_params or {}),
                "n": 1,
                "max_tokens": max_tokens,
                "logprobs": 0 if return_logprobs else None,
            },
        }
        if images:
            payload["content_parts"] = [
                {"type": "image_url", "url": image} for image in images
            ]

        response = await self.client.post(
            "/inference/v1/generate",
            json=payload,
            headers={"X-Session-ID": session_id} if session_id else None,
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]

        detokenized = await self.client.post(
            "/detokenize",
            json={"model": self.model, "tokens": choice["token_ids"]},
        )
        detokenized.raise_for_status()

        return GenerationResult(
            text=detokenized.json()["prompt"],
            token_ids=choice["token_ids"],
            token_logprobs=(
                [item["logprob"] for item in choice["logprobs"]["content"]]
                if return_logprobs
                else None
            ),
            finish_reason=choice["finish_reason"],
        )

    async def tokenize(
        self,
        prompt: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        response = await self.client.post(
            "/tokenize",
            json={
                "model": self.model,
                "prompt": prompt,
                "add_special_tokens": add_special_tokens,
            },
        )
        response.raise_for_status()
        return response.json()["tokens"]

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""

        await self.client.aclose()

    async def __aenter__(self) -> VLLMCompletionBackend:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
