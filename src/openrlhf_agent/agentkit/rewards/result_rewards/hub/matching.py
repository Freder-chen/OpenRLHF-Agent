"""Reward strategy that matches predictions directly against labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from openrlhf_agent.utils.types import Action, Conversation
from openrlhf_agent.agentkit.rewards.result_rewards.base import ResultRewardStrategy
from openrlhf_agent.agentkit.rewards.result_rewards.hub.math_utils import grade_answer_verl


@dataclass
class MatchingReward(ResultRewardStrategy):
    """Default reward strategy that compares predictions and labels."""

    correct_score: float = 1.0
    miss_score: float = 0.0

    def score_response(self, response: str, label: Optional[Any]) -> float:
        """Score a plain-text response against the target label."""

        if label is None:
            return self.miss_score

        target_label = str(label).strip()
        prediction = response.strip()
        return self.correct_score if prediction == target_label else self.miss_score

    async def score(
        self,
        *,
        action: Action,
        label: Optional[Any],
        history: Optional[Conversation] = None,
    ) -> float:
        """Derive a reward from the parsed assistant action."""

        if label is None:
            return self.miss_score

        final_response = self.extract_final_response(action)
        if not final_response:
            return self.miss_score

        return self.score_response(final_response, label)


@dataclass
class MathMatchingReward(MatchingReward):
    """Matching reward that also checks symbolic math equivalence for boxed LaTeX answers."""

    def score_response(self, response: str, label: Optional[Any]) -> float:
        if label is None:
            raise NotImplementedError("label=None is not supported.")

        candidate_labels: Sequence[str]
        if isinstance(label, str):
            candidate_labels = [label]
        elif isinstance(label, Sequence):
            candidate_labels = list(label)
        else:
            raise NotImplementedError(f"Unsupported label type: {type(label)!r}")
    
        response_text = response.strip()
        for gold_label in candidate_labels:
            try:
                if grade_answer_verl(response_text, gold_label):
                    return self.correct_score
            except Exception:
                # Be robust to parser/sympy failures on individual labels.
                continue

        return self.miss_score
