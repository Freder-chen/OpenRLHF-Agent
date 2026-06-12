# OpenRLHF-Agent

> Consistent training and inference stack for building tool-using chat agents on OpenRLHF and vLLM.

OpenRLHF-Agent is a slim runtime for tool-using chat agents. It keeps environment orchestration, chat protocols, and model I/O identical across RL training and production inference.

## Highlights

- **Training = inference**: the same `AgentSession` drives tool calls, transcript rendering, and rewards in both phases.
- **Token-in-token-out**: direct token concatenation with no re-tokenization — avoids BPE mismatch issues.
- **Context compaction**: `CompactableSession` lets the model summarize long conversations and continue from a compressed context.
- **Small surface area**: `AgentSession`, `Environment`, `ChatProtocol`, `LLMEngine`, `AgentRuntime` — easy to audit and extend.

## Architecture

```
AgentRuntime (inference)          agent_func (training)
 │                                 │
 ├─ prompt_ids management          ├─ OpenRLHF manages tokens
 │                                 │
 └─ AgentSession                   └─ AgentSession
     ├─ Conversation                    ├─ Conversation
     ├─ Environment (tools, step)       ├─ Environment
     ├─ ChatProtocol (render, parse)    ├─ ChatProtocol
     └─ RewardPipeline                  └─ RewardPipeline
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full module layout and data flow.

## Quick Start

### Install

```bash
git clone https://github.com/OpenRLHF/OpenRLHF-Agent.git
cd OpenRLHF-Agent
pip install -e .
```

### Run Inference

Start a vLLM endpoint:

```bash
vllm serve Qwen/Qwen3-4B --port 8009 --served-model-name qwen3
```

Run the demo:

```bash
python examples/qwen3/runtime_demo.py
```

### Train with OpenRLHF

```bash
# See examples/qwen3/train_reinforce_agent.sh
# agent_func.py exposes AgentInstance/AgentExecutor for OpenRLHF
```

### Compact (Long Context)

For tasks that exceed the context window, use `CompactableSession`:

```python
from openrlhf_agent.agentkit.session import CompactableSession
from openrlhf_agent.agentkit.runtime import AgentRuntime

session = CompactableSession(
    environment=env,
    protocol=protocol,
)
rt = AgentRuntime(engine, session, max_context_tokens=96000)
```

The runtime automatically compresses context when the token count exceeds the threshold — the model generates a summary, then continues from the compressed state.

See `examples/compact/` for full demos.

## Extend

| Want to... | Do this |
|---|---|
| Add a tool | Subclass `ToolBase`, pass to `FunctionCallEnvironment(tools=[...])` |
| Add a reward | Implement `ResultRewardStrategy` or `ProcessRewardStrategy` |
| Support a new model format | Subclass `ChatProtocol` |
| Add a backend | Implement `LLMEngine` (`generate`, `chat`, `tokenize`) |
| Enable compaction | Use `CompactableSession` + `max_context_tokens` |

## Examples

| Directory | Description |
|---|---|
| `examples/qwen3/` | Multi-turn function calling (inference + training) |
| `examples/search_r1/` | Wiki search agent with retriever |
| `examples/single_turn/` | Single-turn QA |
| `examples/compact/` | Context compaction demo |

## License

Apache License 2.0.
