"""
Minimal agent runtime for OpenRLHF-style tooling.

Only the core entry points are re-exported here; import from submodules for
advanced customization.
  - `orchestrator` hosts AgentRuntime/AgentSession orchestration.
  - `environment` defines the default think/final environment & helpers.
  - `chat_protocol` provides the Qwen3 prompt codec used in examples.
  - `engine` wraps language-model backends (currently an OpenAI-compatible engine).
"""

__version__ = "0.0.1"

from openrlhf_agent.chat_protocol import make_chat_protocol
from openrlhf_agent.environment import make_environment
from openrlhf_agent.engine import OpenAIEngine
from openrlhf_agent.orchestrator import AgentRuntime, AgentSession, AgentStepResult

__all__ = [
    "__version__",
    "AgentRuntime",
    "AgentSession",
    "AgentStepResult",
    "make_environment",
    "make_chat_protocol",
    "OpenAIEngine",
]
