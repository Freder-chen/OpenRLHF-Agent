# OpenRLHF-Agent

> Consistent training and inference stack for building tool-using chat agents on OpenRLHF and vLLM.

OpenRLHF-Agent is a slim runtime for tool-using chat agents. It keeps environment orchestration, chat protocols, and model I/O identical across RL training and production inference.

## Highlights

- **Training = inference**: the same `AgentSession` drives tool calls, transcript rendering, and rewards in both phases.
- **Token-in-token-out**: direct token concatenation with no re-tokenization — avoids BPE mismatch issues.
- **Environment-controlled execution**: environments own tool dispatch, step limits, and termination.
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
python examples/math/runtime_demo.py
```

### Train with OpenRLHF

```bash
# Math
bash examples/math/train_reinforce_agent.sh

# Search with Qwen2.5 Instruct
bash examples/search/qwen2.5_instruct/train_reinforce_agent.sh

# Search with Qwen3 Thinking
bash examples/search/qwen3_thinking/train_reinforce_agent.sh
```

Each example's adjacent `agent_func.py` exposes `AgentInstance` and
`AgentExecutor` for OpenRLHF.

## Extend

| Want to... | Do this |
|---|---|
| Add a tool | Subclass `ToolBase`, pass to `FunctionCallEnvironment(tools=[...])` |
| Add a reward | Implement `ResultRewardStrategy` or `ProcessRewardStrategy` |
| Support a new model format | Subclass `ChatProtocol` |
| Add a backend | Implement `LLMEngine` (`generate`, `tokenize`) |

## Examples

| Directory | Description |
|---|---|
| `examples/math/` | Single-turn math inference, evaluation, and training |
| `examples/search/qwen2.5_instruct/` | Qwen2.5 Instruct search inference, evaluation, and training |
| `examples/search/qwen3_thinking/` | Qwen3 Thinking search inference, evaluation, and training |
| `examples/search/local_dense_retriever/` | Local Wiki retriever setup and server |

## License

Apache License 2.0.
