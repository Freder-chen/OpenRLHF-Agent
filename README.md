# OpenRLHF-Agent

OpenRLHF-Agent is a small runtime for running and training tool-using agents with vLLM, SGLang, and OpenAI-compatible APIs.

## Install

OpenRLHF-Agent requires Python 3.10 or newer.

```bash
git clone https://github.com/Freder-chen/OpenRLHF-Agent.git
cd OpenRLHF-Agent
python -m pip install -e .
```

## Quick Start

Start a compatible vLLM server, then run the math example:

```bash
vllm serve Qwen/Qwen3-4B \
  --port 8009 \
  --served-model-name qwen3

python -m examples.math.runtime_demo
```

## Backends

| Backend | API |
|---|---|
| `VLLMCompletionBackend` | vLLM token-in/token-out API |
| `SGLangCompletionBackend` | SGLang native generation API |
| `OpenAIChatBackend` | Chat Completions API |
| `OpenAIResponsesBackend` | Responses API |

## Examples

| Example | Purpose |
|---|---|
| [`examples/math/`](examples/math/) | Math inference, evaluation, and training |
| [`examples/search/`](examples/search/) | Search-agent inference, evaluation, and training |
| [`examples/robot/`](examples/robot/) | Vision-language robot control in LIBERO |

See [Architecture](docs/ARCHITECTURE.md) for component responsibilities and rollout flow.

## License

Apache License 2.0.
