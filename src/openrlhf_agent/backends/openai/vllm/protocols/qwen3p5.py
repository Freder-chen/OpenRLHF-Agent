"""Qwen3.5 prompt rendering and output parsing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from openrlhf_agent.utils.types import Action, ToolCall

from .base import Protocol


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(?P<name>[^>\s]+)>\s*"
    r"(?P<body>.*?)\s*</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_PARAMETER_RE = re.compile(
    r"<parameter=(?P<name>[^>\s]+)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)


class Qwen3p5Protocol(Protocol):
    """Protocol for the Qwen3.5 nested-tag tool format."""

    chat_template = (Path(__file__).parent / "jinja/qwen3p5.jinja").read_text(encoding="utf-8")

    def __init__(self, *, enable_thinking: bool = True) -> None:
        self.enable_thinking = enable_thinking
        self.template_kwargs = {"enable_thinking": enable_thinking}

    def parse_action(self, text: str) -> Action:
        """Parse optional hybrid thinking followed by a final or tool calls."""

        # Split hidden reasoning from the action.
        reasoning: str | None = None
        if self.enable_thinking:
            end_match = re.search(r"</think>", text, re.IGNORECASE)
            if not end_match:
                return Action(
                    reasoning_content=text.strip() or None,
                    error="Missing </think> tag. Close the reasoning block before acting.",
                )
            reasoning = text[: end_match.start()].strip() or None
            text = text[end_match.end() :].lstrip()

        # Parse the visible action.
        content = text.strip()
        matches = list(_TOOL_CALL_RE.finditer(text))
        if not matches:
            if "<tool_call>" in content.lower():
                return Action(
                    reasoning_content=reasoning,
                    error="Malformed tool call. Use nested function and parameter tags.",
                )
            return Action(content=content or None, reasoning_content=reasoning)

        prefix = text[: matches[0].start()].strip()
        suffix = text[matches[-1].end() :].strip()
        has_text_between_calls = any(
            text[left.end() : right.start()].strip()
            for left, right in zip(matches, matches[1:])
        )
        if "<tool_call>" in prefix.lower() or suffix or has_text_between_calls:
            return Action(
                reasoning_content=reasoning,
                error="Malformed tool call or unexpected text between tool calls.",
            )
        if prefix:
            reasoning = f"{reasoning}\n{prefix}" if reasoning else prefix

        calls: list[ToolCall] = []
        for position, match in enumerate(matches, 1):
            arguments: dict[str, Any] = {}
            body = match.group("body")
            cursor = 0
            for parameter in _PARAMETER_RE.finditer(body):
                if body[cursor : parameter.start()].strip():
                    return Action(
                        reasoning_content=reasoning,
                        error=f"Invalid tool call #{position}: text outside <parameter> tags.",
                    )
                key = parameter.group("name")
                if key in arguments:
                    return Action(
                        reasoning_content=reasoning,
                        error=f"Invalid tool call #{position}: duplicate parameter {key}.",
                    )
                value = parameter.group("value").strip()
                try:
                    arguments[key] = json.loads(value)
                except json.JSONDecodeError:
                    arguments[key] = value
                cursor = parameter.end()

            if body[cursor:].strip():
                return Action(
                    reasoning_content=reasoning,
                    error=f"Invalid tool call #{position}: text outside <parameter> tags.",
                )
            calls.append(
                ToolCall(
                    call_id=f"call_{uuid4().hex}",
                    name=match.group("name"),
                    arguments=arguments,
                )
            )

        return Action(tool_calls=calls, reasoning_content=reasoning)
