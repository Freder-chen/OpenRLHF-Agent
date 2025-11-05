"""Qwen3 chat template rendering and parsing helpers."""

import json
import re
from textwrap import dedent
from typing import ClassVar, List

from openrlhf_agent.core import ParsedAssistantAction, ToolCall
from openrlhf_agent.template.base import Template


# Mirrors the upstream Qwen3 chat formatting rules.
QWEN3_CHAT_TEMPLATE = dedent("""\
{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
            {%- endif %}
        {%- endif %}
        {%- set reasoning_text = reasoning_content.strip('\\n') if reasoning_content else '' %}
        {%- if loop.index0 > ns.last_query_index and reasoning_text %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_text + '\\n</think>\\n\\n' + content.lstrip('\\n') }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}
""").strip()


# Extracts the payload inside the <tool_call> tags.
QWEN3_TOOL_CALL_REGEX = re.compile(
    r"(?P<open><tool_call>)\s*(?P<body>.*?)\s*(?P<close></tool_call>)",
    re.DOTALL | re.IGNORECASE,
)


class Qwen3InstructTemplate(Template):
    """Render Qwen3 messages and parse tool call annotations."""

    chat_template: ClassVar[str] = QWEN3_CHAT_TEMPLATE  # for render_messages
    tool_call_regex: ClassVar[re.Pattern[str]] = QWEN3_TOOL_CALL_REGEX  # for parse_assistant_text

    def parse_assistant_text(self, text: str) -> ParsedAssistantAction:
        """Split assistant text into final content and tool calls."""

        raw = text or ""
        content_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        cursor = 0
        for idx, match in enumerate(self.tool_call_regex.finditer(raw), 1):
            start, end = match.span()
            content_parts.append(raw[cursor:start])
            payload = match.group("body").strip()
            tool_calls.append(self._parse_call(payload, idx=idx))
            cursor = end

        content_parts.append(raw[cursor:])
        content = "".join(content_parts).strip()

        return ParsedAssistantAction(content=content, tool_calls=tool_calls)

    @staticmethod
    def _parse_call(raw_payload: str, idx: int) -> ToolCall:
        """Parse one <tool_call> json blob."""

        try:
            payload = json.loads(raw_payload)
        except Exception as exc:  # simple catch keeps message short
            return ToolCall(id=f"call_{idx}", refusal=f"error parse json: {exc}")

        name = payload.get("name")
        arguments = payload.get("arguments")

        if not isinstance(name, str):
            return ToolCall(id=f"call_{idx}", refusal="error parse json: name must be string.")

        if not isinstance(arguments, dict):
            return ToolCall(id=f"call_{idx}", refusal="error parse json: arguments must be dict.")

        return ToolCall(id=f"call_{idx}", name=name, arguments=arguments)


if __name__ == "__main__":
    template = Qwen3InstructTemplate()
    demo_messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Give me one interesting fact about Mars."},
        {
            "role": "assistant",
            "content": "Let me query the knowledge base for a quick Mars fact.",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "get_mars_fact", "arguments": {"topic": "volcanoes"}},
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"fact": "Olympus Mons is the tallest volcano in the solar system."}',
        },
        # {"role": "assistant", "content": "Mars has the largest volcano in the solar system, Olympus Mons."},
    ]
    rendered = template.render_messages(messages=demo_messages, add_generation_prompt=True)
    print(rendered)
