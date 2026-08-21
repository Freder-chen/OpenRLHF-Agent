"""Reward strategy that proxies a GRM-style external evaluator."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

from openai import AsyncOpenAI

from openrlhf_agent.agentkit.rewards.result_rewards.base import ResultReward
from openrlhf_agent.utils.types import Action, Message


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


@dataclass
class GRMJudgeReward(ResultReward):
    """Reward scored by querying an external GRM-compatible endpoint."""

    model: str
    api_key: str
    base_url: str | None = None

    prompt_template: str = CRITIC_PROMPT_TEMPLATE

    correct_score: float = 1.0
    incorrect_score: float = 0.0
    unknown_score: float = 0.5

    def __post_init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    async def score(
        self,
        *,
        action: Action,
        label: Any,
        question: Sequence[Message] = (),
    ) -> float:
        """Request the external evaluator to score the final answer."""

        if label is None:
            raise ValueError("label is required to score a response")

        response = self.extract_final_response(action)
        if not response:
            return self.incorrect_score

        question_parts = []
        for message in question:
            if message.content is not None and not isinstance(message.content, str):
                raise TypeError("GRMJudgeReward only supports text messages")
            lines = [f"[{message.role.capitalize()}]", message.content or ""]
            for call in message.tool_calls or []:
                arguments = json.dumps(call.arguments or {}, ensure_ascii=False)
                lines.append(f"Tool call: {call.name}({arguments})")
            question_parts.append("\n".join(lines))

        prompt = self.prompt_template.format(
            question="\n\n".join(question_parts),
            label=label,
            response=response,
        )
        try:
            reply = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as error:
            logger.warning("GRM judge request failed: %s", error)
            return self.unknown_score

        if not reply.choices or not reply.choices[0].message.content:
            return self.unknown_score

        verdicts = VERDICT_PATTERN.findall(reply.choices[0].message.content)
        if not verdicts:
            return self.unknown_score
        return (
            self.correct_score
            if verdicts[-1].lower() == "yes"
            else self.incorrect_score
        )
