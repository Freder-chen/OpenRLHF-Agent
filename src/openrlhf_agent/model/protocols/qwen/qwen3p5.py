"""Qwen3.5 prompt rendering and output parsing."""

from pathlib import Path

from openrlhf_agent.model.protocols.base import CompletionProtocol
from openrlhf_agent.utils.types import Action

from .common import parse_nested_tool_action


class Qwen3p5Protocol(CompletionProtocol):
    """Protocol for the Qwen3.5 nested-tag tool format."""

    chat_template = (Path(__file__).parent / "templates" / "qwen3p5.jinja").read_text(
        encoding="utf-8"
    )
    supports_multimodal = True

    def __init__(
        self,
        *,
        enable_thinking: bool = True,
        add_vision_id: bool = False,
    ) -> None:
        self.enable_thinking = enable_thinking
        super().__init__(
            template_kwargs={
                "enable_thinking": enable_thinking,
                "add_vision_id": add_vision_id,
            }
        )

    def parse_action(self, text: str) -> Action:
        """Parse optional reasoning followed by a final answer or tool calls."""

        return parse_nested_tool_action(text, enable_thinking=self.enable_thinking)
