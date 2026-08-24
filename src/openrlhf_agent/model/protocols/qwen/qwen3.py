"""Qwen3 prompt rendering and output parsing."""

from pathlib import Path

from openrlhf_agent.model.protocols.base import CompletionProtocol
from openrlhf_agent.utils.types import Action

from .common import parse_json_tool_action


class Qwen3Protocol(CompletionProtocol):
    """Qwen3 protocol with optional thinking output."""

    chat_template = (Path(__file__).parent / "templates" / "qwen3.jinja").read_text(
        encoding="utf-8"
    )

    def __init__(self, *, enable_thinking: bool) -> None:
        self.enable_thinking = enable_thinking
        super().__init__(template_kwargs={"enable_thinking": enable_thinking})

    def parse_action(self, text: str) -> Action:
        """Parse optional reasoning followed by a final answer or tool calls."""

        return parse_json_tool_action(text, enable_thinking=self.enable_thinking)
