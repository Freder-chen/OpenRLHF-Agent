# OpenRLHF-Agent Architecture

OpenRLHF-Agent separates model communication from agent behavior:

```text
model input -> model output -> Action -> Environment -> new Messages
```

- Backends communicate with model servers.
- Environments define agent behavior and tools.
- Protocols translate text for completion models only.
- The runtime repeats the interaction until the environment finishes.
- Rewards score actions during training.

## Shared Types

| Type | Meaning |
|---|---|
| `Message` | One system, user, assistant, or tool message |
| `ToolCall` | A tool name, arguments, and call ID |
| `Action` | Assistant text, tool calls, reasoning, or a parse error |
| `Observation` | Messages and state returned after an action |
| `Conversation` | Ordered message history |

Backends, environments, and rewards communicate through these types instead of depending on one another's implementations.

## Model Backends

The project supports two model interfaces:

| Backend | Input | Output | Protocol |
|---|---|---|---|
| `ChatBackend` | Structured messages | `Action` | Not needed |
| `CompletionBackend` | Text or token IDs | Token IDs and text | Required |

`OpenAIChatBackend` and `OpenAIResponsesBackend` convert structured messages directly to their API formats.

`VLLMCompletionBackend` uses vLLM's Completions, tokenization, and returned-token-ID extensions. Its protocol converts between shared messages and model-specific text.

vLLM can also serve Chat Completions. In that case, use a chat backend. The distinction is the API being called, not the server product.

## Completion Protocols

A protocol has two operations:

```python
prompt = protocol.render(messages=messages, tools=tools, add_generation_prompt=True)
action = protocol.parse_action(generated_text)
```

Built-in protocols are:

- `Qwen3Protocol(enable_thinking=True | False)` for JSON tool calls.
- `Qwen3p5Protocol(enable_thinking=True | False)` for nested function and parameter tags.

Their templates live in `backends/openai/vllm/protocols/jinja/`. The Python files contain configuration and output parsing.

## Agent Loop

An `Environment` owns the system prompt, tools, step count, and optional step limit. `reset()` starts a rollout. `step(action)` returns `(messages, done)`. A `None` step limit means unlimited steps.

`AgentRuntime` runs one of two paths.

### Completion path

```text
AgentSession.initialize
  -> Protocol.render
  -> CompletionBackend.generate
  -> Protocol.parse_action
  -> Environment.step
  -> render tool feedback
  -> repeat
```

`AgentSession` owns the conversation, protocol, environment transitions, and optional rewards. `AgentRuntime` and the backend own token IDs. Between steps, the runtime appends generated token IDs and tokenizes only new environment feedback.

### Chat path

```text
ChatBackend.generate_chat
  -> Action
  -> Environment.step
  -> append assistant and tool messages
  -> repeat
```

This path keeps structured messages throughout, so it does not need `AgentSession` or a completion protocol. Images remain in the conversation and are sent on every turn.

## Environments and Tools

`FunctionCallEnvironment` runs valid tool calls concurrently. Invalid output and tool failures become messages so the model can retry. A plain-text answer ends the rollout.

`SingleTurnEnvironment` has no tools and ends after one assistant response.

Each `Tool` declares its name, description, and JSON parameter schema, and implements:

```python
async def call(self, arguments: dict[str, Any]) -> Any:
    ...
```

## Rewards

`RewardPipeline` applies process rewards to intermediate steps and result rewards to the final step, then sums the scores. A configured pipeline requires a label.

Built-in rewards cover malformed tools, selected tool penalties, direct matching, math matching, search-answer matching, and external GRM judging.

## Training with OpenRLHF

OpenRLHF owns generation and token accumulation. The example `AgentInstance` only connects it to `AgentSession`:

```text
reset -> initialize with the raw question -> return rendered prompt
step  -> parse action -> run environment -> return reward and feedback
```

The dataset question must remain unformatted because `AgentSession.initialize()` applies the protocol template. Do not enable OpenRLHF's `--data.apply_chat_template` for these agents, or the prompt will be formatted twice.

## Extension Points

| Goal | Extend |
|---|---|
| Add a tool | `Tool` |
| Add an environment | `Environment` |
| Add a process or result reward | `ProcessReward` or `ResultReward` |
| Add a completion format | `Protocol` |
| Add a completion API | `CompletionBackend` |
| Add a chat API | `ChatBackend` |
