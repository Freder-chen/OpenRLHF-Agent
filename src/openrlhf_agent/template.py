import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .types import ParsedAssistantMessage, ToolCall

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

QWEN3_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)

QWEN3_SYSTEM_WITH_TOOLS_TEMPLATE = """
{system}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
""".strip()


class Template(ABC):
    @abstractmethod
    def render_system(self, text: str, tools_manifest: Optional[List[Dict[str, Any]]] = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def render_turn(self, role: str, text: str, *, add_generation_prompt: bool = False) -> str:
        raise NotImplementedError

    @abstractmethod
    def render_generation_prompt(self, role: str = "assistant") -> str:
        raise NotImplementedError

    @abstractmethod
    def render_tool_response(self, text: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def render_messages(
        self,
        *,
        messages: List[Dict[str, str]],
        tools_manifest: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def parse_assistant_message(self, text: str) -> ParsedAssistantMessage:
        raise NotImplementedError


class Qwen3Template(Template):
    def render_generation_prompt(self, role: str = "assistant") -> str:
        return f"{IM_START}{role}\n"

    def render_turn(self, role: str, text: str, *, add_generation_prompt: bool = False) -> str:
        turn = f"{IM_START}{role}\n{text}{IM_END}\n"
        if add_generation_prompt:
            turn += self.render_generation_prompt()
        return turn

    def render_system(
        self,
        text: str,
        tools_manifest: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        rendered = text
        if tools_manifest:
            rendered = QWEN3_SYSTEM_WITH_TOOLS_TEMPLATE.format(
                system=text,
                tools="\n".join([json.dumps(item, ensure_ascii=False) for item in tools_manifest]),
            )
        return self.render_turn(role="system", text=rendered)

    def render_tool_response(self, text: str) -> str:
        return f"<tool_response>\n{text}\n</tool_response>"

    def render_messages(
        self,
        *,
        messages: List[Dict[str, str]],
        tools_manifest: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
    ) -> str:
        blocks: List[str] = []
        pending = list(messages)

        if pending and pending[0]["role"] == "system":
            system_msg = pending.pop(0)
            blocks.append(
                self.render_system(
                    system_msg["content"],
                    tools_manifest=tools_manifest,
                )
            )

        for item in pending:
            blocks.append(self.render_turn(item["role"], item.get("content", "")))

        if add_generation_prompt:
            blocks.append(self.render_generation_prompt())

        return "".join(blocks)

    def parse_assistant_message(self, text: str) -> ParsedAssistantMessage:
        raw = text or ""
        matches = list(QWEN3_TOOL_RE.finditer(raw))

        tool_calls: List[Optional[ToolCall]] = []
        parse_error = False

        cursor = 0

        for idx, match in enumerate(matches):
            start, end = match.span()
            segment = raw[cursor:start]
            if segment.strip():
                parse_error = True

            payload = match.group(1)
            try:
                obj = json.loads(payload)
                name = obj.get("name")
                if not isinstance(name, str) or not name:
                    parse_error = True
                    tool_calls.append(None)
                    cursor = end
                    continue

                raw_arguments = obj.get("arguments", {})
                if isinstance(raw_arguments, dict):
                    arguments_str = json.dumps(raw_arguments or {}, ensure_ascii=False)
                elif isinstance(raw_arguments, str):
                    arguments_str = raw_arguments
                    try:
                        json.loads(arguments_str or "{}")
                    except Exception:
                        parse_error = True
                else:
                    parse_error = True
                    tool_calls.append(None)
                    cursor = end
                    continue

                call_id = obj.get("id") or f"call_{idx}"

                try:
                    tool_calls.append(
                        ToolCall(
                            id=call_id,
                            call_id=call_id,
                            name=name,
                            type="function_call",
                            arguments=arguments_str,
                        )
                    )
                except Exception:
                    parse_error = True
                    tool_calls.append(None)
                    cursor = end
                    continue
            except Exception:
                parse_error = True
                tool_calls.append(None)
            cursor = end

        trailing = raw[cursor:]
        final_response = trailing.strip() or None

        if not matches and not final_response:
            # No tool calls or final response captured – treat as parse error.
            parse_error = True

        return ParsedAssistantMessage(
            parse_error=parse_error,
            tool_calls=tool_calls,
            final_response=final_response,
        )


def make_template(name: Optional[str] = None) -> Template:
    if name in (None, "", "qwen3"):
        return Qwen3Template()
    raise ValueError(f"Unknown template '{name}'.")


__all__ = [
    "Template",
    "Qwen3Template",
    "make_template",
]
