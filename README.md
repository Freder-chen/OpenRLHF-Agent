# OpenRLHF-Agent

OpenRLHF-Agent is a small runtime for training and running tool-using agents.

It provides the same environments, tools, and message types for:

- reinforcement learning with OpenRLHF;
- token-based inference through vLLM Completions;
- chat inference through OpenAI-compatible Chat Completions or Responses APIs.

## Install

OpenRLHF-Agent requires Python 3.10 or newer.

```bash
git clone https://github.com/Freder-chen/OpenRLHF-Agent.git
cd OpenRLHF-Agent
python -m pip install -e .
```

For training, install the OpenRLHF integration instead:

```bash
python -m pip install -e ".[openrlhf]"
```

## Quick Start

Start a vLLM server:

```bash
vllm serve Qwen/Qwen3-4B \
  --port 8009 \
  --served-model-name qwen3
```

Run the math example from the repository root:

```bash
python examples/math/runtime_demo.py
```

The example creates a `VLLMCompletionBackend`, a `Qwen3Protocol`, and a `SingleTurnEnvironment`, then runs them through `AgentRuntime`.

## Choose a Backend

Choose by API endpoint, not by server name:

| Endpoint | Backend | Protocol |
|---|---|---|
| `/v1/completions` with vLLM token IDs | `VLLMCompletionBackend` | Required |
| `/v1/chat/completions` | `OpenAIChatBackend` | Not needed |
| `/v1/responses` | `OpenAIResponsesBackend` | Not needed |

A vLLM server can expose Chat Completions. Use `OpenAIChatBackend` for that endpoint; use `VLLMCompletionBackend` only when the completion and token-ID extensions are needed.

## Examples

| Directory | Purpose |
|---|---|
| [`examples/math/`](examples/math/) | Math inference, evaluation, and training |
| [`examples/search/`](examples/search/) | Qwen2.5 and Qwen3 search examples with local retrieval |
| [`examples/robot/`](examples/robot/) | Multimodal robot inference |

## Train with OpenRLHF

Each training directory contains an `agent_func.py` adapter and a launch script:

```bash
bash examples/math/train_reinforce_agent.sh
bash examples/search/qwen2p5_instruct/train_reinforce_agent.sh
bash examples/search/qwen3_thinking/train_reinforce_agent.sh
```

## Extend

| Goal | Extend |
|---|---|
| Add a tool | `Tool` |
| Add an environment | `Environment` |
| Add a reward | `ProcessReward` or `ResultReward` |
| Add a completion format | `Protocol` |
| Add a model API | `CompletionBackend` or `ChatBackend` |

See [Architecture](docs/ARCHITECTURE.md) for the component boundaries and request flows.

## License

Apache License 2.0.
