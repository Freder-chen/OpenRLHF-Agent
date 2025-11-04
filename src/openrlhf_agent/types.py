from dataclasses import dataclass
from typing import List, Optional

from openai.types.responses import ResponseFunctionToolCallItem


# Re-export the canonical OpenAI function-call tool schema for intra-runtime use.
ToolCall = ResponseFunctionToolCallItem


@dataclass
class ParsedAssistantMessage:
    """Structured output from template parsing of an assistant turn."""

    parse_error: bool
    tool_calls: List[Optional[ToolCall]]
    final_response: Optional[str]
