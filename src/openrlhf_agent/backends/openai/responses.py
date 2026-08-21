"""Backend for the OpenAI Responses API."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from openai import AsyncOpenAI

from openrlhf_agent.backends.base import ChatBackend
from openrlhf_agent.utils.types import Action, ToolCall


class OpenAIResponsesBackend(ChatBackend):
    """Generate actions through OpenAI's Responses API."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: float = 600.0,
        reasoning_effort: str = "low",
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

    async def generate_chat(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Action:
        # Convert chat history into Responses API input items.
        input_items: List[Dict[str, Any]] = []
        for message in messages:
            role = message["role"]

            content = message.get("content")
            converted_content: Any = content
            if isinstance(content, list):
                converted_content = []
                for part in content:
                    if part["type"] == "text":
                        converted_content.append(
                            {"type": "input_text", "text": part["text"]}
                        )
                    elif part["type"] == "image_url":
                        converted_content.append(
                            {"type": "input_image", "image_url": part["image_url"]["url"]}
                        )
                    else:
                        raise ValueError(f"Unsupported content type: {part['type']}")
            elif content is not None and not isinstance(content, str):
                raise TypeError("Message content must be text or a content list.")

            if role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": converted_content or "",
                })
                continue

            if content is not None:
                input_items.append({"role": role, "content": converted_content})
            if role == "assistant":
                for call in message.get("tool_calls", []):
                    input_items.append({
                        "type": "function_call",
                        "call_id": call["call_id"],
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                    })

        # Build the Responses API request.
        request: Dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "reasoning": {"effort": self.reasoning_effort},
        }
        if tools:
            request.update({
                "tools": [{"type": "function", **tool["function"]} for tool in tools],
                "tool_choice": self.tool_choice,
            })

        # Send the request.
        response = await self.client.responses.create(**request)

        # Convert response items into the shared Action format.
        content = []
        reasoning = []
        errors = []
        calls: List[ToolCall] = []
        for item in response.output:
            if item.type == "function_call":
                try:
                    arguments = json.loads(item.arguments)
                except json.JSONDecodeError:
                    return Action(error=f"Malformed tool arguments for {item.name}: {item.arguments}")
                calls.append(ToolCall(call_id=item.call_id, name=item.name, arguments=arguments))
            elif item.type == "message":
                for part in item.content:
                    if part.type == "output_text":
                        content.append(part.text)
                    elif part.type == "refusal":
                        content.append(part.refusal)
            elif item.type == "reasoning":
                reasoning.extend(part.text for part in item.content or item.summary)

        # Preserve incomplete responses as errors for the caller to handle.
        if response.status == "incomplete":
            errors.append("Generation stopped before the response was complete.")
        return Action(
            content="\n".join(content) or None,
            tool_calls=calls or None,
            reasoning_content="\n".join(reasoning) or None,
            error="\n".join(errors) or None,
        )
