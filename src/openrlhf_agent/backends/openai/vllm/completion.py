"""vLLM backend using its OpenAI-compatible Completions API."""

from __future__ import annotations

import httpx

from openrlhf_agent.backends.base import CompletionBackend
from openrlhf_agent.backends.openai.vllm.protocols import Protocol


class VLLMCompletionBackend(CompletionBackend):
    """Generate text and token IDs through vLLM-specific endpoints."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        protocol: Protocol,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.protocol = protocol
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/").removesuffix("/v1"),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
            timeout=timeout,
        )

    async def generate(
        self,
        prompt: str | list[int],
        max_tokens: int | None = None,
    ) -> tuple[list[int], str]:
        response = await self.client.post(
            "/v1/completions",
            json={
                "model": self.model,
                "prompt": prompt,
                # None lets vLLM use the remaining context window.
                "max_tokens": max_tokens,
                "return_token_ids": True,
            },
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        return choice["token_ids"], choice["text"]

    async def tokenize(self, prompt: str) -> list[int]:
        response = await self.client.post(
            "/tokenize",
            json={
                "model": self.model,
                "prompt": prompt,
                "add_special_tokens": True,
            },
        )
        response.raise_for_status()
        return response.json()["tokens"]
