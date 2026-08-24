"""OpenAI Chat Completions and Responses API backends."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import TracebackType
from typing import Any, Sequence

from openai import AsyncOpenAI

from openrlhf_agent.model.backends.base import ActionBackend
from openrlhf_agent.utils.types import Action, ToolCall


def _image_url(part: Mapping[str, Any]) -> tuple[Any, Any]:
    if "image" in part:
        payload = part["image"]
    elif "image_url" in part:
        payload = part["image_url"]
    elif part.get("type") in {"image", "image_url", "input_image"}:
        payload = part.get("url")
    else:
        raise ValueError(f"Unsupported content type: {part.get('type')!r}")

    detail = part.get("detail")
    if isinstance(payload, Mapping):
        detail = payload.get("detail", detail)
        payload = payload.get("url")
    return payload, "auto" if detail is None else detail


def _to_chat_content(
    content: str | list[dict[str, Any]] | None,
) -> str | list[dict[str, Any]] | None:
    if not isinstance(content, list):
        return content

    converted: list[dict[str, Any]] = []
    for part in content:
        if part["type"] in {"text", "input_text"}:
            converted.append({"type": "text", "text": part["text"]})
        else:
            url, detail = _image_url(part)
            converted.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url, "detail": detail},
                }
            )
    return converted


def _to_responses_content(
    content: str | list[dict[str, Any]] | None,
) -> str | list[dict[str, Any]] | None:
    if not isinstance(content, list):
        return content

    converted: list[dict[str, Any]] = []
    for part in content:
        if part["type"] in {"text", "input_text"}:
            converted.append({"type": "input_text", "text": part["text"]})
        else:
            url, detail = _image_url(part)
            converted.append(
                {
                    "type": "input_image",
                    "image_url": url,
                    "detail": detail,
                }
            )
    return converted


class OpenAIChatBackend(ActionBackend):
    """Generate actions through OpenAI's Chat Completions API."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 600.0,
        extra_body: Mapping[str, Any] | None = None,
        tool_choice: Any = "auto",
    ) -> None:
        self.model = model
        self.extra_body = dict(extra_body) if extra_body is not None else {}
        self.tool_choice = tool_choice
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    async def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> Action:
        # Convert the shared message format to Chat Completions messages.
        request_messages: list[dict[str, Any]] = []
        for source in messages:
            message = source.copy()
            if isinstance(message.get("content"), list):
                message["content"] = _to_chat_content(message["content"])

            if calls := message.get("tool_calls"):
                message["tool_calls"] = [
                    {
                        "id": call["call_id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(
                                call.get("arguments") or {},
                                ensure_ascii=False,
                                allow_nan=False,
                            ),
                        },
                    }
                    for call in calls
                ]
            request_messages.append(message)

        request: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "max_completion_tokens": max_tokens,
            "extra_body": self.extra_body,
        }
        if tools:
            request.update(
                {
                    "tools": tools,
                    "tool_choice": self.tool_choice,
                }
            )

        response = await self.client.chat.completions.create(**request)
        message = response.choices[0].message

        content = message.content or message.refusal
        reasoning = getattr(message, "reasoning", None) or getattr(
            message, "reasoning_content", None
        )

        calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                return Action(
                    content=content,
                    reasoning_content=reasoning,
                    error=(
                        f"Malformed tool arguments for {call.function.name}: "
                        f"invalid JSON: {call.function.arguments}"
                    ),
                )
            calls.append(
                ToolCall(
                    call_id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                )
            )

        return Action(
            content=content,
            tool_calls=calls or None,
            reasoning_content=reasoning,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""

        await self.client.close()

    async def __aenter__(self) -> OpenAIChatBackend:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


class OpenAIResponsesBackend(ActionBackend):
    """Generate actions through OpenAI's Responses API."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 600.0,
        reasoning_effort: str | None = None,
        tool_choice: Any = "auto",
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.tool_choice = tool_choice
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    async def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> Action:
        input_items: list[dict[str, Any]] = []
        for message in messages:
            content = _to_responses_content(message.get("content"))
            if message["role"] == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message["tool_call_id"],
                        "output": content or "",
                    }
                )
                continue

            if content is not None:
                input_items.append({"role": message["role"], "content": content})
            input_items.extend(
                {
                    "type": "function_call",
                    "call_id": call["call_id"],
                    "name": call["name"],
                    "arguments": json.dumps(
                        call.get("arguments") or {},
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                }
                for call in message.get("tool_calls") or []
            )

        request: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": max_tokens,
        }
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}
        if tools:
            request.update(
                {
                    "tools": [
                        {**tool["function"], "type": "function"} for tool in tools
                    ],
                    "tool_choice": self.tool_choice,
                }
            )

        output = (await self.client.responses.create(**request)).output
        content_messages: list[str] = []
        reasoning_items: list[str] = []
        calls: list[ToolCall] = []
        error = None
        for item in output:
            if item.type == "reasoning":
                reasoning_text = "".join(
                    part.text for part in (item.content or item.summary)
                )
                if reasoning_text:
                    reasoning_items.append(reasoning_text)
            elif item.type == "message":
                content_messages.append(
                    "".join(
                        part.text if part.type == "output_text" else part.refusal
                        for part in item.content
                        if part.type in {"output_text", "refusal"}
                    )
                )
            elif item.type == "function_call":
                try:
                    arguments = json.loads(item.arguments)
                except json.JSONDecodeError:
                    error = (
                        f"Malformed tool arguments for {item.name}: "
                        f"invalid JSON: {item.arguments}"
                    )
                    continue
                calls.append(
                    ToolCall(
                        call_id=item.call_id,
                        name=item.name,
                        arguments=arguments,
                    )
                )

        return Action(
            content="\n".join(content_messages) or None,
            tool_calls=None if error else calls or None,
            reasoning_content="\n".join(reasoning_items) or None,
            error=error,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""

        await self.client.close()

    async def __aenter__(self) -> OpenAIResponsesBackend:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
