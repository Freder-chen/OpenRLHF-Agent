"""Qwen3 prompt rendering and output parsing."""

import json
import re
from pathlib import Path
from uuid import uuid4

from openrlhf_agent.utils.types import Action, ToolCall

from .base import Protocol


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


class Qwen3Protocol(Protocol):
    """Qwen3 protocol with optional thinking output."""

    chat_template = (Path(__file__).parent / "jinja/qwen3.jinja").read_text(encoding="utf-8")

    def __init__(self, *, enable_thinking: bool) -> None:
        self.enable_thinking = enable_thinking
        self.template_kwargs = {"enable_thinking": enable_thinking}

    def parse_action(self, text: str) -> Action:
        """Parse optional reasoning followed by a final answer or tool calls."""

        # Split hidden reasoning from the action.
        reasoning: str | None = None
        if self.enable_thinking:
            end = text.lower().find("</think>")
            if end == -1:
                return Action(
                    reasoning_content=text.strip() or None,
                    error="Missing </think> tag. Keep the thought short and continue.",
                )

            reasoning = text[:end].strip() or None
            text = text[end + len("</think>") :].lstrip()

        # Parse the visible action.
        content = text.strip()
        matches = list(_TOOL_CALL_RE.finditer(text))
        if not matches:
            if "<tool_call>" in content.lower():
                return Action(
                    reasoning_content=reasoning,
                    error="Malformed tool call. Close the <tool_call> tag.",
                )
            return Action(content=content or None, reasoning_content=reasoning)

        if _TOOL_CALL_RE.sub("", content).strip():
            return Action(
                reasoning_content=reasoning,
                error="Unexpected text outside <tool_call> tags.",
            )

        calls: list[ToolCall] = []
        for position, match in enumerate(matches, 1):
            try:
                payload = json.loads(match.group("body"))
            except json.JSONDecodeError as exc:
                return Action(
                    reasoning_content=reasoning,
                    error=f"Invalid tool call #{position}: {exc}",
                )

            if (
                not isinstance(payload, dict)
                or not isinstance(payload.get("name"), str)
                or not isinstance(payload.get("arguments"), dict)
            ):
                return Action(
                    reasoning_content=reasoning,
                    error=f"Invalid tool call #{position}: expected string name and object arguments.",
                )

            calls.append(
                ToolCall(
                    call_id=f"call_{uuid4().hex}",
                    name=payload["name"],
                    arguments=payload["arguments"],
                )
            )

        return Action(tool_calls=calls, reasoning_content=reasoning)
