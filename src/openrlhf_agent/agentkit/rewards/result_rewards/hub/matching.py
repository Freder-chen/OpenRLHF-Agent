"""Reward strategy that matches predictions directly against labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from openrlhf_agent.utils.types import Action, RewardSample
from openrlhf_agent.agentkit.rewards.result_rewards.base import ResultRewardStrategy
from openrlhf_agent.agentkit.rewards.result_rewards.hub.utils.math import grade_answer_verl
from openrlhf_agent.agentkit.rewards.result_rewards.hub.utils.search import (
    extract_final_answer,
    normalize_answer,
)


@dataclass
class MatchingReward(ResultRewardStrategy):
    """Default reward strategy that compares predictions and labels."""

    correct_score: float = 1.0
    format_score: float = 0.1
    miss_score: float = 0.0

    def score_response(self, response: str, label: Optional[Any]) -> float:
        """Score a plain-text response against the target label."""
        if label is None:
            return self.miss_score

        label = str(label).strip()
        prediction = response.strip()
        return self.correct_score if prediction == label else self.format_score

    async def score(
        self,
        *,
        action: Action,
        label: Optional[Any],
        sample: Optional[RewardSample] = None,
    ) -> float:
        """Derive a reward from the parsed assistant action."""
        assert label is not None

        response = self.extract_final_response(action)
        if not response:
            return self.miss_score

        return self.score_response(response, label)


@dataclass
class MathMatchingReward(MatchingReward):
    """Matching reward that also checks symbolic math equivalence for boxed LaTeX answers."""

    def score_response(self, response: str, label: Optional[Any]) -> float:
        if label is None:
            raise NotImplementedError("label=None is not supported.")

        labels: Sequence[str]
        if isinstance(label, str):
            labels = [label]
        elif isinstance(label, list):
            labels = label
        else:
            raise NotImplementedError(f"Unsupported label type: {type(label)!r}")
    
        resp = response.strip()
        for gold in labels:
            try:
                if grade_answer_verl(resp, gold):
                    return self.correct_score
            except Exception:
                # Be robust to parser/sympy failures on individual labels.
                continue

        return self.format_score


@dataclass
class SearchMatchingReward(MatchingReward):
    """Exact-match reward for Search-R1 style QA: extract the `Answer:` line,
    normalize (lowercase, strip punctuation/articles), and match against any label."""

    def score_response(self, response: str, label: Optional[Any]) -> float:
        if isinstance(label, str):
            labels = [label]
        elif isinstance(label, list):
            labels = label
        else:
            raise NotImplementedError(f"Unsupported label type: {type(label)!r}")

        pred_answer = extract_final_answer(response)
        if pred_answer is None:
            return self.miss_score

        normalized_labels = {normalize_answer(str(item)) for item in labels}
        normalized_pred_answer = normalize_answer(pred_answer)
        return self.correct_score if normalized_pred_answer in normalized_labels else self.format_score
