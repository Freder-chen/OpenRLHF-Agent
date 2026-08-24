"""Qwen3.6 prompt rendering and output parsing."""

from pathlib import Path

from openrlhf_agent.model.protocols.base import CompletionProtocol
from openrlhf_agent.utils.types import Action

from .common import parse_nested_tool_action


class Qwen3p6Protocol(CompletionProtocol):
    """Protocol for the Qwen3.6 nested-tag tool format."""

    chat_template = (Path(__file__).parent / "templates" / "qwen3p6.jinja").read_text(
        encoding="utf-8"
    )
    supports_multimodal = True

    def __init__(
        self,
        *,
        enable_thinking: bool = True,
        preserve_thinking: bool = False,
        add_vision_id: bool = False,
    ) -> None:
        self.enable_thinking = enable_thinking
        super().__init__(
            template_kwargs={
                "enable_thinking": enable_thinking,
                "preserve_thinking": preserve_thinking,
                "add_vision_id": add_vision_id,
            }
        )

    def parse_action(self, text: str) -> Action:
        """Parse optional reasoning followed by a final answer or tool calls."""

        return parse_nested_tool_action(text, enable_thinking=self.enable_thinking)
