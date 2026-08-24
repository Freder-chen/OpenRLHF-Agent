"""Prompt rendering and action parsing for completion-model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Any, NoReturn

from jinja2 import Environment, Template

from openrlhf_agent.utils.types import Action


def _raise_exception(message: str) -> NoReturn:
    """Expose a predictable error helper to checkpoint-owned Jinja templates."""

    raise ValueError(message)


_JINJA_ENV = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
_JINJA_ENV.globals["raise_exception"] = _raise_exception
_JINJA_ENV.policies["json.dumps_kwargs"] = {
    **_JINJA_ENV.policies.get("json.dumps_kwargs", {}),
    "ensure_ascii": False,
    "sort_keys": False,
}


@lru_cache(maxsize=None)
def _compile_template(source: str) -> Template:
    """Compile and cache chat templates keyed by their source."""

    return _JINJA_ENV.from_string(source)


@dataclass(slots=True)
class RenderedPrompt:
    """Text prompt and image payloads produced from the same message traversal."""

    text: str
    images: list[Any] = field(default_factory=list)


def _collect_images(
    messages: Sequence[Mapping[str, Any]],
) -> list[Any]:
    """Collect image payloads in content-part order."""

    images: list[Any] = []

    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                # TODO: Add video support to RenderedPrompt.
                if (
                    part.get("type") in {"video", "video_url", "input_video"}
                    or "video" in part
                    or "video_url" in part
                ):
                    raise ValueError(
                        "Video content is not supported; "
                        "RenderedPrompt only transports images."
                    )

                if "image" in part:
                    image = part["image"]
                elif "image_url" in part:
                    image = part["image_url"]
                    if isinstance(image, Mapping):
                        image = image["url"]
                elif part.get("type") == "image":
                    raise ValueError("Image content requires image or image_url.")
                else:
                    continue

                images.append(image)

    return images


class CompletionProtocol(ABC):
    """Render structured messages and parse raw completion-model output."""

    chat_template: str
    supports_multimodal = False

    def __init__(self, *, template_kwargs: Mapping[str, Any] | None = None) -> None:
        self.template_kwargs: Mapping[str, Any] = MappingProxyType(
            dict(template_kwargs or {})
        )

    def render(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedPrompt:
        """Render messages and keep their ordered image payloads beside the text."""

        images = _collect_images(messages)
        if images and not self.supports_multimodal:
            raise ValueError(
                f"{type(self).__name__} does not support multimodal completion prompts."
            )

        text = _compile_template(self.chat_template).render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            **self.template_kwargs,
        )
        return RenderedPrompt(text=text, images=images)

    @abstractmethod
    def parse_action(self, text: str) -> Action:
        """Parse generated assistant text into an action."""
