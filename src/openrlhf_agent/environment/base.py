"""Shared interfaces for agent environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from openrlhf_agent.core import ParsedAssistantAction, ToolCall


class Environment(ABC):
    """Abstract interface for agent environments."""

    @property
    @abstractmethod
    def max_steps(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def tools_manifest(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def execute_tool(self, call: ToolCall, context: Dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def reward_hook(self, action: ParsedAssistantAction, label: Optional[str]) -> float:
        raise NotImplementedError

    @abstractmethod
    def reset_step(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        action: ParsedAssistantAction,
        label: Optional[str] = None,
        runtime: bool = False,
    ) -> Tuple[List[str], float, bool, Optional[str]]:
        raise NotImplementedError
