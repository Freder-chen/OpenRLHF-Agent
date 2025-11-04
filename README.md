# OpenRLHF-Agent

> Consistent training and inference stack for building tool-using chat agents on OpenRLHF and vLLM.

OpenRLHF-Agent provides a shared runtime that covers environment orchestration, prompt templates, and model I/O for both RL training and production inference. Teams can prototype an agent policy with OpenRLHF, then ship the same codepath behind a chatbot powered by vLLM or any OpenAI-compatible endpoint.

## ✨ Highlights

- **Training and inference stay aligned**: the identical `AgentSession` flow drives environment resets, tool calls, and transcript rendering in both phases.
- **Agent-first primitives**:
  - `Environment`: tracks rewards, tool availability, and end conditions.
  - `Template`: renders prompts, parses `<tool_call>` records, and summarizes turns.
  - `LLMEngine`: wraps OpenAI APIs, vLLM servers, or custom backends.
- **Tool-centric design**: bundled `think` and `final` tools demonstrate ReAct-style loops out of the box.
- **Production-ready samples**: Qwen-3 examples cover inference serving, RL data collection, and REINFORCE++ training.
- **Optimized for OpenRLHF**: plug `AgentRuntime` into `train_reinforce_agent.sh` or Ray jobs without extra glue code.

## 🧭 Why this matters in 2025

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

Key folders:

- `src/openrlhf_agent/environment/`: default environments and tool definitions.
- `src/openrlhf_agent/template/`: prompt builders, response parsers, and session serializers.
- `src/openrlhf_agent/runtime/`: runtime loop, engines, and shared session helpers.
- `examples/qwen3/`: runnable Qwen-3 demos for inference and reinforcement learning.

## 🚀 Quick start

### 1. Install

```bash
git clone https://github.com/OpenRLHF/OpenRLHF-Agent.git
cd OpenRLHF-Agent
pip install -e .
# optional developer extras
pip install -e .[dev]
```

Optional vLLM support:

```bash
pip install vllm>=0.10.2
```

### 2. Launch vLLM (optional)

See `examples/qwen3/run_vllm.sh`:

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

This demo uses:

- `OpenAIEngine` to call a vLLM OpenAI-compatible endpoint.
- `DefaultEnvironment` with the `think` and `final` tools.
- `Qwen3Template` to manage `<tool_call>` payloads and transcripts.

You will see tool traces and the final answer in the console.

### 4. Integrate with OpenRLHF training

Check `examples/qwen3/agent_func.py` for the training bridge:

1. `AgentInstance.reset` prepares the system prompt, tool hints, and environment state.
2. `AgentInstance.step` decodes tool calls, updates rewards, and yields `AgentStepResult`.
3. `train_reinforce_agent.sh` provides a `ray job submit` example (set `DATASET_PATH`).

## 🛠️ Customize the stack

### Add a tool

1. Subclass `environment/tools/base.py::ToolBase`.
2. Implement `call(self, context, **kwargs)` to return visible output or JSON.
3. Register it from your environment during initialization.

### Build a custom environment

- Override `reward_hook` for domain-specific scoring.
- Extend `step` to orchestrate multiple tool calls or enforce safety checks.
- Use `_internal_obs` to provide hidden hints to the model between turns.

### Ship a new prompt template

- Subclass `template/base.py::Template`.
- Implement render and parse helpers for your prompt style.
- Register it with `make_template` before constructing the runtime.

### Support another engine

- Subclass `runtime/engine/base.py::LLMEngine`.
- Implement `generate` and `tokenize` for your provider.
- Pass the engine into `AgentRuntime` or custom entry points.

## 🌅 Project vision

OpenRLHF-Agent is the open-source bridge between RLHF-style training loops and production-grade agent deployments. By aligning tool schemas, prompts, and environment contracts, it lowers the barrier for teams that want to:

- Train agents with reward-driven planning, self-monitoring, and safety checks.
- Deploy those agents behind proactive chat products without reimplementing logic.
- Experiment with emerging agent patterns (long-term memory, hierarchical planners, multi-agent collaboration) while keeping a maintainable codebase.

## 📂 Repository layout

```
OpenRLHF-Agent/
├── src/openrlhf_agent/
│   ├── environment/
│   ├── runtime/
│   ├── template/
│   └── utils/
├── examples/qwen3/
├── pyproject.toml
├── setup.py
└── README.md
```

## 🤝 Contributing

1. Fork and clone the repo.
2. Install dev dependencies: `pip install -e .[dev]`.
3. Run `ruff`, `mypy`, and `pytest` before submitting.
4. Confirm the Qwen-3 training and inference demos still run.
5. Open a PR summarizing the change and test results.

## 📄 License

Apache License 2.0.
