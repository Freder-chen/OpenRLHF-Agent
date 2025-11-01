# OpenRLHF-Agent

> Training–inference consistent Agent SDK for OpenRLHF & vLLM.

OpenRLHF-Agent gives you a clear and editable agent framework. The same code handles both training (with OpenRLHF) and inference (with vLLM), so you do not need to maintain two different versions.

## ✨ Highlights

- **Training and inference stay aligned**: Environments, tool formats, and prompts are shared in both phases.
- **Modular structure**:
  - `Environment`: holds tools, rewards, and stop rules.
  - `Template`: builds prompts and reads `<tool_call>` blocks.
  - `LLMEngine`: talks to the model service (OpenAI API, vLLM, or custom).
- **Tool-first design**: default `think` and `final` tools make the flow easy to trace.
- **Ready-to-run samples**: Qwen-3 demos for inference and RL training.
- **Tight OpenRLHF integration**: see `examples/qwen3/agent_func.py` to plug into training loops.

## 🧱 How it fits together

```
AgentRuntime
 ├─ Template      (renders prompts, parses tool calls)
 ├─ Environment   (manages state, rewards, tools)
 └─ LLMEngine     (generates tokens via OpenAI/vLLM/custom)
```

Key folders:

- `src/openrlhf_agent/environment/`: default environment and tools.
- `src/openrlhf_agent/template/`: prompt builders and parsers.
- `src/openrlhf_agent/runtime/`: runtime loop plus engines.
- `examples/qwen3/`: scripts for Qwen-3 models.

## 🚀 Quick start

### 1. Install

```bash
git clone https://github.com/OpenRLHF/OpenRLHF-Agent.git
cd OpenRLHF-Agent
pip install -e .
# optional developer tools
pip install -e .[dev]
```

Optional vLLM support:

```bash
pip install vllm>=0.10.2
```

### 2. Launch vLLM (optional)

Example from `examples/qwen3/run_vllm.sh`:

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

- `OpenAIEngine` for talking to vLLM’s OpenAI-compatible endpoint.
- `DefaultEnvironment` with the `think` and `final` tools.
- `Qwen3Template` to handle `<tool_call>` blocks.

You will see the tool steps and final answer in the console.

### 4. Use it in OpenRLHF training

`examples/qwen3/agent_func.py` shows the full integration:

1. `AgentInstance.reset` builds the system prompt and tool hints.
2. `AgentInstance.step` parses tool calls and runs environment rewards.
3. `train_reinforce_agent.sh` contains a `ray job submit` example (set `DATASET_PATH` yourself).

## 🛠️ Customize it

### Add a new tool

1. Subclass `environment/tools/base.py::ToolBase`.
2. Implement `call(self, context, **kwargs)` to return user-visible text or JSON.
3. Register it in a custom environment (e.g., inside `__init__`).

### Create a custom environment

- Override `reward_hook` for your reward logic.
- Change `step` to handle multi-tool calls or special stops.
- Use `_internal_obs` to send hidden hints to the model.

### Support a new prompt template

- Subclass `template/base.py::Template`.
- Implement render/parse methods.
- Register it via `make_template`.

### Plug in another engine

- Subclass `runtime/engine/base.py::LLMEngine`.
- Implement `generate` and `tokenize`.
- Pass your engine when creating `AgentRuntime`.

## 📂 Project layout

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
4. Make sure training and inference demos still work.
5. Open a PR with your changes and notes.

## 📄 License

Apache License 2.0.
