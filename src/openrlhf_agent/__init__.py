"""
Minimal agent runtime for OpenRLHF-style tooling.

Only the core entry points are re-exported here; import from submodules for
advanced customization.
  - `agent` hosts AgentRuntime/AgentSession orchestration.
  - `environment` defines the default think/final environment & helpers.
  - `template` provides the Qwen3 chat template used in examples.
  - `model` wraps language-model backends (currently an OpenAI-compatible engine).
"""

__version__ = "0.0.1"

from .agent import AgentRuntime, AgentSession
from .environment import make_environment
from .model import OpenAIEngine
from .template import make_template

__all__ = [
    "__version__",
    "AgentRuntime",
    "AgentSession",
    "make_environment",
    "make_template",
    "OpenAIEngine",
]
