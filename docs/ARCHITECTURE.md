# OpenRLHF Agent Architecture

OpenRLHF-Agent uses the same environment, protocol, session, and reward primitives for OpenRLHF rollouts and inference against an OpenAI-compatible model server.

## Module Layout

```text
src/openrlhf_agent/
├── utils/types/
│   ├── conversation.py    # Message, ToolCall, Conversation
│   └── action.py          # Action, Observation, RewardSample
├── backends/
│   ├── base.py            # LLMEngine interface: generate, tokenize
│   └── hub/openai.py      # OpenAI/vLLM completions and tokenization client
└── agentkit/
    ├── runtime.py          # Inference loop and token management
    ├── session.py          # Message state, environment transitions, rewards
    ├── environments/
    │   ├── base.py         # Tools, system prompt, step counter, max steps
    │   └── hub/            # FunctionCallEnvironment, SingleTurnEnvironment
    ├── protocols/
    │   ├── base.py         # ChatProtocol rendering/parsing interface
    │   └── hub/            # Qwen3InstructProtocol, Qwen3ThinkingProtocol
    ├── tools/
    │   ├── base.py         # ToolBase and OpenAI tool manifest generation
    │   └── hub/            # Control, Wiki search, and Jina tools
    └── rewards/
        ├── pipeline.py     # Process/final reward orchestration
        ├── process_rewards/  # ToolFormatReward
        └── result_rewards/   # Matching, math, search, and GRM rewards
```

## Core Components

### Environment

`Environment` owns the system prompt, registered tools, current step, and maximum step count. Its transition contract is:

```python
observations, done = await environment.step(action)
```

`FunctionCallEnvironment` executes valid tool calls concurrently. Parse, validation, and tool runtime failures are returned as structured JSON tool observations so the model can retry. A plain-text assistant response terminates the episode, and reaching `max_steps` forces termination.

`SingleTurnEnvironment` has no tools and always terminates after one action.

### Protocol

`ChatProtocol` converts structured messages and tool manifests into model input, then parses generated text into an `Action`. The built-in protocols support:

- `Qwen3InstructProtocol` for non-reasoning Qwen tool-call transcripts.
- `Qwen3ThinkingProtocol` for transcripts containing a `<think>...</think>` reasoning section.

A response may contain either a plain-text final answer or tool calls. Text mixed with tool calls in the same action is treated as a parse failure.

### Session

`AgentSession` connects the protocol, environment, conversation history, and an optional `RewardPipeline`. It never manages token IDs.

Initialization resets the environment step counter, rebuilds the conversation with the environment system prompt, and renders the initial model prompt. Each step:

1. Parses generated text into an `Action`.
2. Appends the assistant action to the conversation.
3. Applies the action to the environment.
4. Converts tool outputs into feedback messages and rendered feedback text.
5. Computes an optional reward from the action, label, and `RewardSample`.
6. Returns `(Observation, reward)`.

`Observation` contains `step_index`, `feedback_messages`, `feedback_text`, and the boolean `done` flag.

### Runtime

`AgentRuntime` creates an inference-only `AgentSession` from an engine, environment, and protocol. The runtime owns token IDs; the session owns message state.

```text
session.initialize(messages) -> prompt text
engine.tokenize(prompt text)  -> prompt_ids

repeat up to environment.max_steps:
    engine.generate(prompt_ids)         -> action_ids, action_text
    prompt_ids += action_ids
    session.step_from_text(action_text) -> observation, reward
    yield assistant/tool messages
    if observation.done: return
    prompt_ids += tokenize(feedback_text)
```

If the model exhausts its context window or never produces a final answer, the runtime emits a final `Max steps reached without final response.` message.

### Rewards

`RewardPipeline` applies a process reward on non-final tool steps and a result reward on the final step.

- `ToolFormatReward` penalizes malformed tool actions.
- `MatchingReward` performs direct answer matching.
- `MathMatchingReward` supports symbolic/boxed math equivalence.
- `SearchMatchingReward` extracts the final `Answer:` line and performs
  normalized exact matching.
- `GRMJudgeReward` delegates final-answer grading to an external judge model.

Matching rewards distinguish three outcomes: `correct_score`, `format_score`, and `miss_score`.

### Tools

Every tool subclasses `ToolBase`, declares an OpenAI-compatible name, description, and JSON schema, and implements an asynchronous `call` method. Built-in tools include:

- `WikiSearchTool` for the local retriever service.
- `JinaSearchTool` and `JinaReadTool` for Jina APIs.
- `ThinkTool`, `CommentaryTool`, and `FinalTool` control helpers.

## Inference Data Flow

```text
AgentRuntime
    -> LLMEngine.generate
    -> ChatProtocol.parse_assistant_text
    -> AgentSession
    -> Environment
        -> ToolBase.call (when tools are requested)
    -> Observation
    -> rendered tool feedback or final answer
```

The runtime appends generated token IDs directly and tokenizes only newly rendered tool feedback. It does not repeatedly render and tokenize the complete conversation.

## Training Data Flow

OpenRLHF owns generation and rollout token accumulation. An example `AgentInstance` handles only message-level transitions:

```text
reset(states)
    -> session.initialize(observation)
    -> return rendered prompt to OpenRLHF

step(states)
    -> session.step_from_text(action_text, label=label)
    -> return reward, done, and feedback_text to OpenRLHF
```

The adjacent `agent_func.py` in each training example exposes `AgentInstance` and `AgentExecutor`.

## Extending

| Goal | Extension point |
|---|---|
| Add a tool | Subclass `ToolBase` and pass it to `FunctionCallEnvironment` |
| Add an environment | Subclass `Environment` and implement `step` |
| Add a reward | Implement `ResultRewardStrategy` or `ProcessRewardStrategy` |
| Support another model format | Subclass `ChatProtocol` |
| Add a model backend | Implement `LLMEngine.generate` and `LLMEngine.tokenize` |
