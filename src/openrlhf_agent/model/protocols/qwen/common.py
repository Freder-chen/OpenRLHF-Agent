"""Shared action parsers for Qwen completion protocols."""

from __future__ import annotations

import json
import re
from typing import Any
from uuid import uuid4

from openrlhf_agent.utils.types import Action, ToolCall


_JSON_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_NESTED_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(?P<name>[^>\s]+)>\s*"
    r"(?P<body>.*?)\s*</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_PARAMETER_RE = re.compile(
    r"<parameter=(?P<name>[^>\s]+)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)
_THINK_END_RE = re.compile(r"</think>", re.IGNORECASE)


def _split_reasoning(
    text: str,
    *,
    enable_thinking: bool,
    missing_tag_error: str,
) -> tuple[str | None, str] | Action:
    if not enable_thinking:
        return None, text

    end_match = _THINK_END_RE.search(text)
    if end_match is None:
        return Action(
            reasoning_content=text.strip() or None,
            error=missing_tag_error,
        )

    reasoning = text[: end_match.start()].strip() or None
    return reasoning, text[end_match.end() :].lstrip()


def parse_json_tool_action(text: str, *, enable_thinking: bool) -> Action:
    """Parse the JSON tool-call format used by Qwen3."""

    text = text.rstrip().removesuffix("<|im_end|>")
    split = _split_reasoning(
        text,
        enable_thinking=enable_thinking,
        missing_tag_error="Missing </think> tag. Close the reasoning block before acting.",
    )
    if isinstance(split, Action):
        return split
    reasoning, text = split

    content = text.strip()
    matches = list(_JSON_TOOL_CALL_RE.finditer(text))
    if not matches:
        if "<tool_call>" in content.lower():
            return Action(
                reasoning_content=reasoning,
                error="Malformed tool call. Close the <tool_call> tag.",
            )
        return Action(content=content or None, reasoning_content=reasoning)

    if _JSON_TOOL_CALL_RE.sub("", content).strip():
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
                error=(
                    f"Invalid tool call #{position}: expected string name and object arguments."
                ),
            )

        calls.append(
            ToolCall(
                call_id=f"call_{uuid4().hex}",
                name=payload["name"],
                arguments=payload["arguments"],
            )
        )

    return Action(tool_calls=calls, reasoning_content=reasoning)


def parse_nested_tool_action(text: str, *, enable_thinking: bool) -> Action:
    """Parse the nested function/parameter format used since Qwen3.5."""

    text = text.rstrip().removesuffix("<|im_end|>")
    split = _split_reasoning(
        text,
        enable_thinking=enable_thinking,
        missing_tag_error="Missing </think> tag. Close the reasoning block before acting.",
    )
    if isinstance(split, Action):
        return split
    reasoning, text = split

    content = text.strip()
    matches = list(_NESTED_TOOL_CALL_RE.finditer(text))
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

    return Action(
        content=prefix or None,
        tool_calls=calls,
        reasoning_content=reasoning,
    )
