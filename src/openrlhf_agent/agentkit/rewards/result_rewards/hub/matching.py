"""Reward strategy that matches predictions directly against labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from openrlhf_agent.utils.types import Action, Message
from openrlhf_agent.agentkit.rewards.result_rewards.base import ResultReward
from openrlhf_agent.agentkit.rewards.result_rewards.hub.utils.math import answers_match
from openrlhf_agent.agentkit.rewards.result_rewards.hub.utils.search import (
    extract_final_answer,
    normalize_answer,
)


@dataclass
class MatchingReward(ResultReward):
    """Default reward strategy that compares predictions and labels."""

    correct_score: float = 1.0
    format_score: float = 0.1
    miss_score: float = 0.0

    def score_response(self, response: str, label: Any) -> float:
        """Score a plain-text response against the target label."""
        if label is None:
            raise ValueError("label is required to score a response")

        label = str(label).strip()
        prediction = response.strip()
        return self.correct_score if prediction == label else self.format_score

    async def score(
        self,
        *,
        action: Action,
        label: Any,
        question: Sequence[Message] = (),
    ) -> float:
        """Derive a reward from the parsed assistant action."""
        if label is None:
            raise ValueError("label is required to score a response")

        response = self.extract_final_response(action)
        if not response:
            return self.miss_score

        return self.score_response(response, label)


class MathMatchingReward(MatchingReward):
    """Matching reward that also checks symbolic math equivalence for boxed LaTeX answers."""

    def score_response(self, response: str, label: Any) -> float:
        if label is None:
            raise ValueError("label is required to score a response")

        labels = [label] if isinstance(label, str) else label
        if not isinstance(labels, list):
            raise TypeError(f"label must be a string or list, got {type(label).__name__}")

        for gold in labels:
            if answers_match(response, gold):
                return self.correct_score

        return self.format_score


class SearchMatchingReward(MatchingReward):
    """Exact-match reward for Search-R1 style QA: extract the `Answer:` line,
    normalize (lowercase, strip punctuation/articles), and match against any label."""

    def score_response(self, response: str, label: Any) -> float:
        if label is None:
            raise ValueError("label is required to score a response")

        labels = [label] if isinstance(label, str) else label
        if not isinstance(labels, list):
            raise TypeError(f"label must be a string or list, got {type(label).__name__}")

        pred_answer = extract_final_answer(response)
        if pred_answer is None:
            return self.miss_score

        normalized_labels = {normalize_answer(str(item)) for item in labels}
        normalized_pred_answer = normalize_answer(pred_answer)
        return (
            self.correct_score
            if normalized_pred_answer in normalized_labels
            else self.format_score
        )
