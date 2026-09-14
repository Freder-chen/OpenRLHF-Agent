# Architecture

OpenRLHF-Agent separates model communication, model-specific formatting, rollout state, and environment behavior.

## Core Components

| Component | Responsibility |
|---|---|
| `AgentRuntime` | Runs the model-environment loop |
| `AgentSession` | Stores conversation history, applies environment actions, and computes optional rewards |
| `Environment` | Defines the system prompt, tools, transitions, step limit, and terminal condition |
| `ActionBackend` | Sends structured messages to a model API and returns an `Action` |
| `CompletionBackend` | Sends token IDs to a completion API and owns its `CompletionProtocol` |
| `CompletionProtocol` | Renders structured messages and parses generated text |

An `Action` is the protocol-independent assistant output: text, reasoning, tool calls, or a parse error. An `Observation` contains the assistant action, environment feedback, step index, and terminal state.

## Rollout Paths

`AgentRuntime` selects one of two paths:

- `ActionBackend` sends structured messages and returns an `Action`. OpenAI Chat Completions and Responses use this path.
- `CompletionBackend` uses its `CompletionProtocol` to render messages into text and ordered images, tokenizes the text, generates exact token IDs, and parses the generated text into an `Action`. vLLM and SGLang use this path.

Both paths pass the `Action` to `AgentSession`. The completion path appends sampled token IDs unchanged and renders only new feedback; generated text is parsed but never retokenized.

## Environments and Rewards

`Environment.reset()` returns the initial messages. `Environment.step(action)` returns feedback messages and `done`. The environment owns its step count and terminal decision.

- `SingleTurnEnvironment` ends after one assistant reply.
- `FunctionCallEnvironment` runs independent tools concurrently and returns errors as feedback so the model can retry.
- `RobotEnvironment` delegates tools to a robot client and runs them sequentially because each action changes the next observation.

`AgentSession` appends each action and its feedback to history. An optional `RewardPipeline` applies process rewards to non-terminal steps and result rewards to the terminal step.
