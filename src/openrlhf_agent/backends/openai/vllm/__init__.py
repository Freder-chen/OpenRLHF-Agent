"""vLLM extensions to the OpenAI Completions API."""

from .completion import VLLMCompletionBackend
from .protocols import Protocol, Qwen3Protocol, Qwen3p5Protocol
