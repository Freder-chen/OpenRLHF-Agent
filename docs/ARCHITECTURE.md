# Architecture

OpenRLHF-Agent separates model communication, model-specific text formatting, and agent behavior. Each component owns one clear part of the rollout.

## Core Components

| Component | Owns |
|---|---|
| `CompletionBackend` | Tokenization, generation requests, token IDs, logprobs, and image transport |
| `ActionBackend` | Structured-message requests and conversion to `Action` |
| `CompletionProtocol` | Checkpoint chat template, ordered image collection, and generated-text parsing |
| `AgentRuntime` | Backend selection, rollout loop, completion token IDs, and accumulated images |
| `AgentSession` | Completion history, protocol rendering, environment transitions, and optional rewards |
| `Environment` | System prompt, tools, step count, observations, and terminal state |

Backends and protocols are siblings under `openrlhf_agent.model`. A backend does not own a protocol: `AgentRuntime` composes them only for the completion path.

## Two Rollout Paths

```text
Action path:
Messages -> ActionBackend.generate -> Action -> Environment.step -> Messages

Completion path:
Messages -> CompletionProtocol.render -> RenderedPrompt
         -> CompletionBackend.generate -> GenerationResult
         -> CompletionProtocol.parse_action -> Action
         -> Environment.step -> Observation
```

`OpenAIChatBackend` and `OpenAIResponsesBackend` use the action path. `VLLMCompletionBackend` and `SGLangCompletionBackend` use the completion path.

### Completion Path

The completion path keeps semantic state and token state separate.

| State | Owner |
|---|---|
| Structured messages and parsed actions | `AgentSession` |
| Exact generated token IDs | `AgentRuntime` and `GenerationResult` |
| Accumulated ordered images | `AgentRuntime` |
| Step count and `done` | `Environment` |

One rollout proceeds as follows:

1. `AgentSession.reset()` combines the environment's initial messages with the question and returns a `RenderedPrompt`.
2. `AgentRuntime` tokenizes the initial prompt once with `add_special_tokens=False` because the chat template already contains its control tokens.
3. The completion backend generates `GenerationResult.text` and the exact sampled `GenerationResult.token_ids`.
4. The runtime appends the sampled token IDs unchanged and passes only the text to `AgentSession.step()` for action parsing and environment execution.
5. `AgentSession.step()` returns `(Observation, reward)`. If the rollout continues, the runtime tokenizes only `Observation.feedback_text` and appends `Observation.environment_images`.
6. The next backend request receives all accumulated token IDs and images.

The generated text is not retokenized. Text is still required because protocols and environments operate on semantic actions rather than token IDs.

Incremental observation rendering currently supports only the bundled Qwen templates. `AgentSession` compares the prompt before and after an observation so historical thinking, turn separators, image numbering, and generation prompts remain consistent. Supporting another template requires adapting this suffix logic; implementing `CompletionProtocol.render()` alone is not enough.

### Action Path

The action path keeps structured messages throughout the rollout. `AgentRuntime` sends the complete conversation and tool manifest to the action backend, appends the returned assistant message, executes the environment, and repeats until `done=True`.

This path does not use `AgentSession` or `CompletionProtocol`. The backend owns provider-specific conversion for Chat Completions or Responses.

## Shared Types

| Type | Meaning |
|---|---|
| `Message` | One system, user, assistant, or tool turn |
| `ToolCall` | Tool name, arguments, call ID, and optional error |
| `Action` | Assistant content, reasoning, tool calls, or a parse error |
| `RenderedPrompt` | Rendered completion text and matching ordered images |
| `GenerationResult` | Generated text, exact token IDs, optional logprobs, finish reason, and backend metadata |
| `Observation` | New messages, completion feedback text, new images, step index, and terminal state |

## Environments, Tools, and Rewards

`Environment.reset()` starts one rollout and returns its initial messages. `Environment.step(action)` executes one transition and returns `(messages, done)`. The environment alone owns its step limit and terminal decision.

`FunctionCallEnvironment` executes valid tool calls concurrently. Parser errors and tool failures become feedback messages so the model can retry; a plain-text answer ends the rollout.

A `Tool` returns either text or an ordered list of text and image content parts. The environment places that result directly into a `Message` without rearranging its parts.

`RewardPipeline` is optional and belongs to `AgentSession`. Non-terminal steps use process rewards; terminal steps use result rewards. `AgentRuntime` does not create a reward pipeline, while the OpenRLHF examples construct `AgentSession` directly when rewards are needed.

## Images

Images may come from the initial messages or from later tool and environment observations.

```text
ordered Message content parts
  -> CompletionProtocol.render
  -> RenderedPrompt(text, images)
  -> AgentRuntime(token IDs, accumulated images)
  -> CompletionBackend
  -> model server and checkpoint processor
```

The completion protocol emits model-specific vision placeholders and collects image payloads in the same order. It does not resize images or create model tensors. The completion backend transports the images, and the model server performs checkpoint-specific visual processing.

Action backends do not create `RenderedPrompt`. They translate ordered content parts directly to the provider's structured image format.

Current limitations:

- video content is not supported;
- vLLM images require a server build compatible with the backend's `content_parts` request;
- the OpenRLHF example adapters return text only and do not forward `RenderedPrompt.images` or `Observation.environment_images`.

## OpenRLHF Examples

Each OpenRLHF `AgentInstance` owns an `AgentSession`. Its `reset()` method returns `RenderedPrompt.text`; its `step()` method passes generated text to the session and returns feedback, terminal state, and reward in OpenRLHF's expected state dictionary.

OpenRLHF owns generation and token accumulation in this path. The shared session continues to own prompt formatting, action parsing, environment transitions, and reward calculation.

## Extension Points

| Goal | Extend |
|---|---|
| Add a tool | `Tool` |
| Add an environment | `Environment` |
| Add a process or result reward | `ProcessReward` or `ResultReward` |
| Add a completion format | `CompletionProtocol` |
| Add a completion API | `CompletionBackend` |
| Add a structured-message API | `ActionBackend` |
