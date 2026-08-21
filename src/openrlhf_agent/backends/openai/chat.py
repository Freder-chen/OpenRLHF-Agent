"""Backend for the OpenAI Chat Completions API."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from openai import AsyncOpenAI

from openrlhf_agent.backends.base import ChatBackend
from openrlhf_agent.utils.types import Action, ToolCall


class OpenAIChatBackend(ChatBackend):
    """Generate actions through OpenAI's Chat Completions API."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: float = 600.0,
        extra_body: Optional[Dict[str, Any]] = None,
        tool_choice: Any = "auto",
    ) -> None:
        self.model = model
        self.extra_body = extra_body or {}
        self.tool_choice = tool_choice
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    async def generate_chat(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Action:
        # Convert chat history into Chat Completions messages.
        request_messages: List[Dict[str, Any]] = []
        for source in messages:
            message = source.copy()
            if "reasoning_content" in message:
                message["reasoning"] = message.pop("reasoning_content")

            if "tool_calls" in message:
                message["tool_calls"] = [
                    {
                        "id": call["call_id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                        },
                    }
                    for call in message["tool_calls"]
                ]
            request_messages.append(message)

        # Build the Chat Completions request.
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "extra_body": self.extra_body,
        }
        if tools:
            request.update({"tools": tools, "tool_choice": self.tool_choice})

        # Send the request and read the assistant message.
        response = await self.client.chat.completions.create(**request)
        choice = response.choices[0]
        message = choice.message

        # Convert OpenAI tool calls into the shared Action format.
        calls = []
        for call in message.tool_calls or []:
            raw_arguments = call.function.arguments
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                return Action(
                    content=message.content,
                    error=f"Malformed tool arguments for {call.function.name}: {raw_arguments}",
                )
            calls.append(ToolCall(
                call_id=call.id,
                name=call.function.name,
                arguments=arguments,
            ))
        reasoning = getattr(message, "reasoning", None)

        # Treat truncated output as a retryable error.
        if choice.finish_reason == "length":
            return Action(
                reasoning_content=reasoning,
                error="Generation stopped before the response was complete.",
            )

        # Return the normalized assistant action.
        return Action(
            content=message.content or message.refusal or None,
            tool_calls=calls or None,
            reasoning_content=reasoning,
        )
