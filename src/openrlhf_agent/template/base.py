"""Base classes and helpers for chat templates."""

from abc import ABC
from typing import Any, ClassVar, Dict, List, Optional, Sequence

from jinja2 import Environment

from openrlhf_agent.core import ParsedAssistantAction


class Template(ABC):
    """Base helper that knows how to render and parse provider chat formats."""

    chat_template: ClassVar[Optional[str]] = None
    _jinja_env: ClassVar[Environment] = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
    _jinja_env.policies["json.dumps_kwargs"] = {**_jinja_env.policies.get("json.dumps_kwargs", {}), "ensure_ascii": False,}
    _compiled_chat_template: ClassVar[Optional[Any]] = None
    _compiled_source: ClassVar[Optional[str]] = None

    @classmethod
    def _get_compiled_template(cls):
        """Return a compiled Jinja template and cache it for reuse."""

        template_source = getattr(cls, "chat_template", None)
        if template_source is None:
            raise NotImplementedError(
                f"{cls.__name__} must define 'chat_template' or override 'render_messages'."
            )

        if cls._compiled_chat_template is not None and cls._compiled_source == template_source:
            return cls._compiled_chat_template

        cls._compiled_chat_template = cls._jinja_env.from_string(template_source)
        cls._compiled_source = template_source
        return cls._compiled_chat_template

    def render_messages(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools_manifest: Optional[Sequence[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
    ) -> str:
        """Render messages using the provider chat template."""

        template = self._get_compiled_template()
        tools = list(tools_manifest) if tools_manifest else None
        return template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )

    def parse_assistant_text(self, text: str) -> ParsedAssistantAction:
        """Turn a raw assistant reply into a structured object."""

        raise NotImplementedError
