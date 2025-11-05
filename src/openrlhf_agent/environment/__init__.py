"""Environment primitives and default factory."""

from __future__ import annotations

from typing import Any, Optional

from openrlhf_agent.environment.base import Environment
from openrlhf_agent.environment.function_call import FunctionCallEnvironment


def make_environment(name: Optional[str] = None, **kwargs: Any) -> Environment:
    """Create an environment by name; returns the default when unspecified."""

    if name in (None, "default"):
        return FunctionCallEnvironment(**kwargs)
    raise ValueError(f"Unknown environment '{name}'.")


__all__ = ["Environment", "FunctionCallEnvironment", "make_environment"]

