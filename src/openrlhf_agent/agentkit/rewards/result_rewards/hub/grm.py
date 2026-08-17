"""Reward strategy that proxies a GRM-style external evaluator."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from jinja2 import Environment

from openai import AsyncOpenAI

from openrlhf_agent.agentkit.rewards.result_rewards.base import ResultRewardStrategy
from openrlhf_agent.utils.types import Action, RewardSample


logger = logging.getLogger(__name__)


CRITIC_PROMPT_TEMPLATE = """
Act as an impartial evaluator to determine whether an AI assistant’s response is consistent with—or exceeds—the quality of a provided reference answer to a given question.
Consider the following factors: helpfulness, relevance, accuracy, depth, creativity, harmlessness, and overall quality. Analyze these dimensions based on the specific problem, as different tasks may emphasize different criteria.
Avoid biases related to the position of responses, response length, or assistant names. Be objective in your assessment.
Output your judgment strictly as: [[Yes]] if the assistant’s response is consistent with or better than the reference answer, [[No]] otherwise.

[User Question]
{question}

[The Start of Reference Answer]
{label}
[The End of Reference Answer]

[The Start of Assistant's Response]
{response}
[The End of Assistant's Response]
""".strip()

VERDICT_PATTERN = re.compile(r"\[\[(Yes|No)\]\]", re.IGNORECASE)

_PROMPT_ENV = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)


def _render_tool_call_payload(payload: Any) -> Optional[str]:
    """Return a compact, human-readable representation of a tool call."""

    if payload is None:
        return None

    if isinstance(payload, Mapping):
        data = dict(payload)
    elif hasattr(payload, "model_dump"):
        data = payload.model_dump(exclude_none=True)
    else:
        return None

    call_id = data.get("call_id") or data.get("id")
    arguments: Any = data.get("arguments")
    name = data.get("name")

    if not name and isinstance(data.get("function"), Mapping):
        function_payload = data["function"]
        name = function_payload.get("name", name)
        if arguments is None:
            arguments = function_payload.get("arguments")

    if isinstance(arguments, str):
        args_text = arguments
    elif arguments is None:
        args_text = "{}"
    else:
        try:
            args_text = json.dumps(arguments, ensure_ascii=False)
        except TypeError:
            args_text = str(arguments)

    lines = ["<tool_call>"]
    if call_id:
        lines.append(f"id: {call_id}")
    if name:
        lines.append(f"name: {name}")
    lines.append(f"arguments: {args_text}")
    lines.append("</tool_call>")
    return "\n".join(lines)


_QUESTION_TEMPLATE = _PROMPT_ENV.from_string(
"""
{% for message in messages -%}
[{{ message.role|capitalize }}]
{{ message.content or "" }}
{% if message.tool_calls %}
{% for tool_call in message.tool_calls -%}
{{ tool_call }}
{% if not loop.last %}

{% endif %}
{% endfor %}
{% endif %}
{% if not loop.last %}

{% endif %}
{% endfor %}
""".strip()
)


def _normalize_messages(payload: Iterable[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for entry in payload:
        if entry is None:
            continue

        if isinstance(entry, Mapping):
            data = dict(entry)
        elif hasattr(entry, "model_dump"):
            data = entry.model_dump(exclude_none=True)
        else:
            continue

        normalized.append(
            {
                "role": data.get("role"),
                "content": data.get("content"),
                "tool_calls": [
                    rendered for rendered in (
                        _render_tool_call_payload(call)
                        for call in data.get("tool_calls") or []
                    ) if rendered
                ] or None,
            }
        )
    return normalized


def render_question_from_sample(sample: Optional[RewardSample]) -> str:
    if not sample or not sample.question:
        return ""

    question_payload = sample.question
    if isinstance(question_payload, str):
        return question_payload.strip()

    formatted_messages = _normalize_messages(question_payload)
    if not formatted_messages:
        return ""

    return _QUESTION_TEMPLATE.render(messages=formatted_messages).strip()


def extract_verdict(text: str) -> Optional[str]:
    """Parse the final [[Yes]] / [[No]] verdict emitted by the judge."""

    if not text:
        return None

    matches = VERDICT_PATTERN.findall(text)
    return matches[-1] if matches else None


@dataclass
class GRMJudgeReward(ResultRewardStrategy):
    """Reward scored by querying an external GRM-compatible endpoint."""

    model: Optional[str]
    base_url: Optional[str]
    api_key: Optional[str]

    prompt_template: str = CRITIC_PROMPT_TEMPLATE

    correct_score: float = 1.0
    format_score: float = 0.0
    error_score: float = -0.1

    def __post_init__(self) -> None:
        self._base_url = self.base_url
        self._api_key = self.api_key
        self._model = self.model

        client_kwargs: dict[str, Any] = {
            "base_url": self._base_url,
            "api_key": self._api_key
        }

        self._client = AsyncOpenAI(**client_kwargs)

    def _prepare_prompt(self, *, question: str, label: str, response: str) -> str:
        return self.prompt_template.format(question=question, label=label, response=response)

    async def _score_with_judge(self, prompt: str) -> Optional[str]:
        """Send the prompt to the external judge model."""

        try:
            reply = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # pragma: no cover - runtime error path
            logger.warning("GRM judge request failed: %s", exc)
            return None

        if not reply.choices:
            return None

        content = reply.choices[0].message.content or ""
        return content.strip() or None

    async def score(
        self,
        *,
        action: Action,
        label: Optional[Any],
        sample: Optional[RewardSample] = None,
    ) -> float:
        """Request the external evaluator to score the final answer."""

        response = self.extract_final_response(action)
        if not response:
            return self.error_score

        question_text = render_question_from_sample(sample)
        prompt = self._prepare_prompt(question=question_text, label=label, response=response)
        verdict_text = await self._score_with_judge(prompt)
        if not verdict_text:
            return self.error_score

        verdict = extract_verdict(verdict_text)
        if verdict is None:
            return self.format_score

        verdict_lower = verdict.lower()
        if verdict_lower == "yes":
            return self.correct_score
        if verdict_lower == "no":
            return self.format_score

        return self.format_score


if __name__ == "__main__":
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
                },
            ],
        },
        {
            "role": "tool",
            "content": '{"fact": "Olympus Mons is the tallest volcano in the solar system."}',
        },
        {"role": "assistant", "content": "Mars has the largest volcano in the solar system, Olympus Mons."},
    ]
    formatted_messages = _normalize_messages(demo_messages)
    question_text = _QUESTION_TEMPLATE.render(messages=formatted_messages).strip()
    print(question_text)
