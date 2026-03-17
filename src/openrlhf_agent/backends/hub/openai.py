"""OpenAI-compatible engine implementation."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import httpx
from openai import AsyncOpenAI

from openrlhf_agent.backends.base import LLMEngine


class OpenAIEngine(LLMEngine):
    """Thin wrapper that talks to an OpenAI-style completions endpoint."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        self.model = model or os.getenv("OPENAI_MODEL")

        # NOTE: Separate client needed for tokenization route.
        base_for_split = (self.base_url or "").rstrip("/")
        tokenize_base_url = "/".join(base_for_split.split("/")[:-1]) if base_for_split else ""
        if not tokenize_base_url:
            tokenize_base_url = base_for_split
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._token_client: Optional[httpx.AsyncClient]
        if tokenize_base_url:
            self._token_client = httpx.AsyncClient(
                base_url=tokenize_base_url,
                headers=headers or None,
            )
        else:
            self._token_client = None

    async def generate(
        self,
        prompt: Optional[Union[str, List[int]]],
        max_tokens: int = 10240,
        temperature: float = 0.6,
    ) -> Tuple[List[int], str]:
        response = await self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={
                "return_token_ids": True,
            },
        )
        token_ids = response.choices[0].token_ids
        text = response.choices[0].text
        return token_ids, text

    async def chat(
        self,
        messages: Sequence[Dict[str, str]],
        max_tokens: int = 10240,
        temperature: float = 0.6,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        message = response.choices[0].message
        text = message.content or ""
        return text if isinstance(text, str) else str(text)

    async def tokenize(self, prompt: str) -> List[int]:
        if self._token_client is None:
            raise RuntimeError("Tokenization client is unavailable; set OPENAI_BASE_URL for this backend.")
        response = await self._token_client.post(
            "/tokenize",
            json={
                "model": self.model,
                "prompt": prompt,
                "add_special_tokens": True,
            },
        )
        response.raise_for_status()
        payload = response.json()
        return payload["tokens"]
