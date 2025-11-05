"""Reward helpers for evaluating final answers."""

from __future__ import annotations

from typing import Optional


def extract_verdict(text: str) -> Optional[str]:
    """Return [[A]] or [[B]] if found; otherwise None."""

    verdicts = [match for match in ("[[A]]", "[[B]]") if match in text]
    return verdicts[-1] if verdicts else None


def compute_reward(
    response: str,
    target: str,
    *,
    correct_score: float = 1.0,
    verdict_score: float = 0.1,
    miss_score: float = 0.0,
) -> float:
    """Score the response against a ground-truth label string."""

    gold = target.strip()
    if response.strip() == gold:
        return correct_score
    if extract_verdict(response) == gold:
        return verdict_score
    return miss_score


__all__ = ["compute_reward", "extract_verdict"]

