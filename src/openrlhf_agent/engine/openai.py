"""OpenAI-compatible engine implementation."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

import httpx
from openai import OpenAI

from openrlhf_agent.engine.base import LLMEngine


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

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        self.model = model or os.getenv("OPENAI_MODEL")

        # Separate client needed for tokenization route.
        self.tmp_client = OpenAI(
            base_url="/".join(self.base_url.split("/")[:-1]),
            api_key=self.api_key,
        )

    def generate(
        self,
        prompt: Optional[Union[str, List[int]]],
        max_tokens: int = 10240,
        temperature: float = 0.6,
        stream: bool = False,
    ) -> Tuple[List[int], str]:
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            extra_body={
                "return_token_ids": True,
            },
        )
        token_ids = response.choices[0].token_ids
        text = response.choices[0].text
        return token_ids, text

    def tokenize(self, prompt: str) -> List[int]:
        response = self.tmp_client.post(
            "/tokenize",
            body=dict(
                model=self.model,
                prompt=prompt,
                add_special_tokens=True,
            ),
            cast_to=httpx.Response,
        ).json()
        return response["tokens"]


__all__ = ["OpenAIEngine"]
