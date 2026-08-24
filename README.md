# OpenRLHF-Agent

OpenRLHF-Agent is a small runtime for running and training tool-using agents. It keeps model-server APIs separate from environments, tools, and rewards.

It provides:

- token-in/token-out backends for vLLM and SGLang;
- OpenAI-compatible Chat Completions and Responses backends;
- Qwen completion templates and tool-call parsers;
- ordered text and image observations for VLMs;
- OpenRLHF training examples.

## Install

OpenRLHF-Agent requires Python 3.10 or newer.

```bash
git clone https://github.com/Freder-chen/OpenRLHF-Agent.git
cd OpenRLHF-Agent
python -m pip install -e .
```

Model servers and example-specific dependencies must be installed separately.

## Quick Start

The math demo uses `VLLMCompletionBackend` and requires a vLLM build that exposes `/inference/v1/generate`, `/tokenize`, and `/detokenize`.

```bash
vllm serve Qwen/Qwen3-4B \
  --port 8009 \
  --served-model-name qwen3

python examples/math/runtime_demo.py
```

The example creates one runtime from a backend, protocol, and environment:

```python
runtime = AgentRuntime(
    backend=VLLMCompletionBackend(
        model="qwen3",
        base_url="http://localhost:8009/v1",
        api_key="empty",
    ),
    protocol=Qwen3Protocol(enable_thinking=True),
    environment=SingleTurnEnvironment(system_prompt="You are a helpful assistant."),
)

answer = await runtime.run_final([{"role": "user", "content": "1+1=?"}])
```

## How It Works

| Component | Responsibility |
|---|---|
| Backend | Communicate with one model API |
| Completion protocol | Render a checkpoint template and parse generated text |
| Environment | Own tools, observations, step count, and termination |
| Runtime | Connect the backend to the environment until the rollout finishes |

Completion backends consume rendered text or token IDs and require a protocol when used by `AgentRuntime`. Action backends consume structured messages directly and do not use a protocol.

## Backends

Choose a backend by API endpoint, not by server product.

| Backend | Endpoint | Use when |
|---|---|---|
| `VLLMCompletionBackend` | `/inference/v1/generate` | You need vLLM-generated token IDs and optional logprobs |
| `SGLangCompletionBackend` | `/generate` | You need SGLang-generated token IDs, optional logprobs, or native image transport |
| `OpenAIChatBackend` | `/v1/chat/completions` | The server accepts structured chat messages and tools |
| `OpenAIResponsesBackend` | `/v1/responses` | The server supports Responses reasoning, tools, and structured input |

A vLLM server can also expose Chat Completions. Use `OpenAIChatBackend` for that endpoint and `VLLMCompletionBackend` for the token-in/token-out endpoint.

## Completion Protocols

| Protocol | Tool-call format | Images |
|---|---|---|
| `Qwen3Protocol` | JSON inside `<tool_call>` | No |
| `Qwen3p5Protocol` | Nested function and parameter tags | Yes |
| `Qwen3p6Protocol` | Nested function and parameter tags | Yes |
| `Qwen3p8Protocol` | Nested function and parameter tags | Yes |

## Examples

| Example | Purpose | Requirements |
|---|---|---|
| [`examples/math/`](examples/math/) | Math inference, evaluation, and OpenRLHF training | Model server; training scripts assume Ray and multiple GPUs |
| [`examples/search/`](examples/search/) | Tool-using search inference, evaluation, and training | Start the [local retriever](examples/search/local_dense_retriever/README.md) first |

The OpenRLHF adapters live in each example's `agent_func.py`. Training launch scripts are starting points and should be adjusted for the available cluster and model.

## Extend

| Goal | Extend |
|---|---|
| Add a tool | `Tool` |
| Add an environment | `Environment` |
| Add a reward | `ProcessReward` or `ResultReward` |
| Add a completion format | `CompletionProtocol` |
| Add a model API | `CompletionBackend` or `ActionBackend` |

See [Architecture](docs/ARCHITECTURE.md) for component ownership, rollout flow, and multimodal boundaries.

## License

Apache License 2.0.
