"""Message encoding and action parsing for completion models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Sequence

from jinja2 import Environment, Template

from openrlhf_agent.utils.types import Action


_JINJA_ENV = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
_JINJA_ENV.policies["json.dumps_kwargs"] = {
    **_JINJA_ENV.policies.get("json.dumps_kwargs", {}),
    "ensure_ascii": False,
    "sort_keys": False,
}


@lru_cache(maxsize=None)
def _compile_template(source: str) -> Template:
    """Compile and cache chat templates keyed by their source."""

    return _JINJA_ENV.from_string(source)


class Protocol(ABC):
    """Render model prompts and parse model output."""

    chat_template: str
    template_kwargs: dict[str, Any] = {}

    def render(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> str:
        """Render structured messages as one completion prompt."""

        return _compile_template(self.chat_template).render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            **self.template_kwargs,
        )

    @abstractmethod
    def parse_action(self, text: str) -> Action:
        """Parse generated assistant text into an action."""
