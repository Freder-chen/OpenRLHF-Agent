# OpenRLHF-Agent

> Consistent training and inference stack for building tool-using chat agents on OpenRLHF and vLLM.

OpenRLHF-Agent provides a shared runtime that covers environment orchestration, prompt templates, and model I/O for both RL training and production inference. Teams can prototype an agent policy with OpenRLHF, then ship the same codepath behind a chatbot powered by vLLM or any OpenAI-compatible endpoint.

## ✨ Highlights

- **Training and inference stay aligned**: the identical `AgentSession` flow drives resets, tool calls, and transcript rendering across phases.
- **Lean agent primitives**: a minimal set of modules (`AgentRuntime`, `Environment`, `Template`, `LLMEngine`, and shared core models) keeps the runtime easy to audit and extend.
- **Tool-centric design**: bundled `think` helper demonstrates ReAct-style loops while final answers ship as plain assistant text.
- **Production-ready examples**: Qwen-3 samples cover inference serving, RL data collection, and REINFORCE++ training.
- **Optimized for OpenRLHF**: plug `AgentRuntime` into `train_reinforce_agent.sh` or Ray jobs without extra glue code.

## 🧭 Why this matters

Chat assistants are shifting from passive Q&A toward autonomous task execution. Leading providers now expose agent modes that plan actions, invoke tools, and maintain long-lived context. OpenRLHF-Agent focuses on the engineering glue needed to keep those behaviors consistent between experimentation and deployment. Use it to:

- Iterate on multi-step reasoning policies with reward shaping and safety hooks before you ship.
- Connect the same prompt strategy to live inference endpoints without rewriting tool logic.
- Extend agents with memory stores, search APIs, or enterprise tools while staying within a single runtime abstraction.

## 🧱 Architecture

```
AgentRuntime
 ├─ AgentSession  (shared rollouts for training + inference)
 ├─ Template      (prompt rendering, tool parsing)
 ├─ Environment   (state, rewards, tool registry)
 └─ LLMEngine     (token streaming via OpenAI/vLLM/custom)
```

### Where the pieces live

- `src/openrlhf_agent/core/`: shared chat, tool-call, and step-result models.
- `src/openrlhf_agent/orchestrator/`: runtime loop, `AgentRuntime`, and `AgentSession` orchestration.
- `src/openrlhf_agent/environment/`: default environment, tool registry, reward hooks, and tool base classes.
- `src/openrlhf_agent/template/`: prompt builders, `<tool_call>` parsing, and template factory helpers.
- `src/openrlhf_agent/engine/`: OpenAI-compatible `LLMEngine` base and the default HTTP client.
- `examples/qwen3/`: runnable demos for inference and reinforcement learning.

## 🚀 Quick start

### 1. Install

```bash
git clone https://github.com/OpenRLHF/OpenRLHF-Agent.git
cd OpenRLHF-Agent
pip install -e .
# optional extras
pip install -e .[dev]        # linting & tests
pip install -e .[openrlhf]   # pulls OpenRLHF core packages
```

Runtime dependencies are also listed in `requirements.txt` if you prefer `pip install -r requirements.txt`.

### 2. (Optional) launch vLLM

Install vLLM (e.g. `pip install vllm>=0.10.2`) and start the Qwen-3 endpoint via `examples/qwen3/run_vllm.sh`, or run it manually:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --port 8009 \
  --served-model-name qwen \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8
```

### 3. Run the inference demo

```bash
python examples/qwen3/runtime_demo.py
```

The script wires together:

- `OpenAIEngine` pointing at a vLLM/OpenAI-compatible endpoint.
- `DefaultEnvironment` with the `think` tool, feedback hooks, and plain-text finals.
- `make_template("qwen3")` for prompt rendering and `<tool_call>` parsing.

You will see tool traces and the final answer printed to the console.

### 4. Plug into OpenRLHF training

`examples/qwen3/agent_func.py` exposes the `AgentInstance` / `AgentExecutor` hooks required by OpenRLHF. Run `examples/qwen3/train_reinforce_agent.sh` (set `DATASET_PATH`) or integrate the functions into your own Ray jobs to collect trajectories and train policies.

## 🛠️ Customize the stack

### Add a tool

1. Subclass `ToolBase` from `environment/tools.py`.
2. Implement `call(self, context, **kwargs)` to return visible output or structured JSON.
3. Register the tool on your environment (`env.registry.register(...)`) before starting the runtime.

### Tailor the environment

- Override `reward_hook(action, label)` for domain-specific scoring.
- Extend `step` to orchestrate multiple tool calls or enforce guardrails.
- Emit hidden hints through `_internal_obs` to steer the policy between turns.

### Ship a new prompt template

- Subclass `Template` in `template/base.py`.
- Implement render + parse helpers for your prompt style.
- Expose it via `make_template` and pass it into `AgentRuntime`.

### Support another engine

- Subclass `LLMEngine` in `engine/base.py`.
- Implement `generate` and `tokenize` for your provider.
- Instantiate the engine and supply it to `AgentRuntime`.

## 🌅 Project vision

OpenRLHF-Agent is the open-source bridge between RLHF-style training loops and production-grade agent deployments. By aligning tool schemas, prompts, and environment contracts, it lowers the barrier for teams that want to:

- Train agents with reward-driven planning, self-monitoring, and safety checks.
- Deploy those agents behind proactive chat products without reimplementing logic.
- Experiment with emerging agent patterns (long-term memory, hierarchical planners, multi-agent collaboration) while keeping a maintainable codebase.

## 🤝 Contributing

1. Fork and clone the repo.
2. Install dev dependencies: `pip install -e .[dev]`.
3. Run `ruff`, `mypy`, and `pytest` (or the demos) before submitting.
4. Confirm the Qwen-3 training and inference demos still run.
5. Open a PR summarizing behaviour changes and test coverage.

## 📄 License

Apache License 2.0.
