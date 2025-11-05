"""Factory helpers for chat templates."""

from typing import Dict, Optional, Type

from openrlhf_agent.template.base import Template
from openrlhf_agent.template.qwen3_instruct import Qwen3InstructTemplate


_DEFAULT_TEMPLATE = "qwen3_instruct"
_TEMPLATE_REGISTRY: Dict[str, Type[Template]] = {
    "qwen3_instruct": Qwen3InstructTemplate,
}


def make_template(name: Optional[str] = None) -> Template:
    """Return a template instance by name."""

    resolved_name = (name or _DEFAULT_TEMPLATE).lower()
    try:
        template_cls = _TEMPLATE_REGISTRY[resolved_name]
    except KeyError as exc:
        raise ValueError(f"Unknown template '{name}'.") from exc

    return template_cls()


__all__ = ["Template", "Qwen3InstructTemplate", "make_template"]
