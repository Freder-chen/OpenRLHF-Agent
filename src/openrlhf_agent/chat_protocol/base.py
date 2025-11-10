"""Base classes and helpers for chat protocols."""

from abc import ABC
from functools import lru_cache
from typing import Any, ClassVar, Dict, List, Optional, Sequence

from jinja2 import Environment, Template

from openrlhf_agent.core import ParsedAssistantAction


_JINJA_ENV = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
_JINJA_ENV.policies["json.dumps_kwargs"] = {
    **_JINJA_ENV.policies.get("json.dumps_kwargs", {}),
    "ensure_ascii": False,
}


@lru_cache(maxsize=None)
def _compile_template(source: str) -> Template:
    """Compile and cache chat templates keyed by their source."""

    return _JINJA_ENV.from_string(source)


class ChatProtocol(ABC):
    """Provider-specific codec for rendering and parsing chat transcripts."""

    chat_template: ClassVar[Optional[str]] = None

    def render_messages(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools_manifest: Optional[Sequence[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
    ) -> str:
        """Render messages using the provider chat template."""

        template_source = self.chat_template
        if template_source is None:
            raise NotImplementedError(
                f"{type(self).__name__} must define 'chat_template' or override 'render_messages'."
            )

        tools = list(tools_manifest) if tools_manifest else None
        template = _compile_template(template_source)
        return template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )

    def parse_assistant_text(self, text: str) -> ParsedAssistantAction:
        """Turn a raw assistant reply into a structured object."""

        raise NotImplementedError
