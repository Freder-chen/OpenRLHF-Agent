# OpenRLHF Agent Architecture

OpenRLHF-Agent uses the same primitives for RL rollouts and production inference: `Environment`, `ChatProtocol`, `AgentSession`, `AgentRuntime`, and provider `LLMEngine`s.

## Module Layout

```
src/openrlhf_agent/
├── utils/types/
│   ├── conversation.py    # Message, ToolCall, Conversation
│   └── action.py          # Action, Status(CONTINUE/DONE), Observation
├── backends/
│   ├── base.py            # LLMEngine interface (generate, chat, tokenize)
│   └── hub/openai.py      # OpenAI/vLLM HTTP client
└── agentkit/
    ├── session/
    │   ├── base.py         # AgentSession — one continuous conversation segment
    │   └── compactable.py  # CompactableSession — adds context compression
    ├── runtime.py          # AgentRuntime — inference loop with token management
    ├── environments/
    │   ├── base.py         # Environment interface (system_prompt, tools, step)
    │   └── hub/            # FunctionCallEnvironment, SingleTurnEnvironment
    ├── tools/
    │   ├── base.py         # ToolBase
    │   └── hub/            # ThinkTool, WikiSearchTool, JinaReadTool, etc.
    ├── protocols/
    │   ├── base.py         # ChatProtocol (render/parse)
    │   └── hub/            # Qwen3ThinkingProtocol, Qwen3InstructProtocol
    └── rewards/
        ├── pipeline.py     # RewardPipeline
        ├── process_rewards/  # Per-step rewards (ToolCallReward)
        └── result_rewards/   # Final-turn rewards (MatchingReward, GRMJudgeReward)
```

## Data Flow

### Inference (AgentRuntime)

```
AgentRuntime
 │
 ├─ session.initialize(messages)  →  prompt_text
 │     └─ tokenize  →  prompt_ids
 │
 └─ loop:
      ├─ engine.generate(prompt_ids)  →  action_ids, action_text
      ├─ session.step_from_text(action_text)  →  observation
      ├─ if DONE: return
      ├─ prompt_ids += action_ids + tokenize(feedback_text)
      └─ if token_count > threshold:
           ├─ session.request_compact()  →  compact_feedback
           ├─ engine.generate(prompt_ids + compact_feedback)  →  summary
           ├─ session.finish_compact(summary)  →  new_prompt
           └─ prompt_ids = tokenize(new_prompt)
```

### Training (agent_func + OpenRLHF)

```
OpenRLHF framework controls generate and token accumulation.
AgentInstance only manages message-level state:

reset(states):
    session.initialize(payload)  →  prompt_text  →  return to framework

step(states):
    session.step_from_text(action_text)  →  observation, reward
    return feedback_text to framework (framework appends tokens)
```

## Key Design Decisions

**Token-in-token-out**: Runtime accumulates `prompt_ids` by direct concatenation. No re-tokenization of prior turns — avoids BPE mismatch issues that arise from `apply_chat_template` on full history.

**Session is message-level only**: `AgentSession` tracks `Conversation` and `step_index`, but never touches token IDs. The caller (runtime or OpenRLHF) owns the token state.

**Compact is two-step**: `request_compact()` returns the compact instruction as feedback_text. The same model generates the summary (tokens stay in the training trajectory). `finish_compact(summary)` re-initializes the session.

**Environment is pure**: Only defines `system_prompt`, `tools`, and `step(action) → (observations, done)`. No step counting, no compaction — those belong to session/caller.

**Status enum**: `Observation.status` is `CONTINUE` or `DONE`. No hidden states.

## Extending

| Want to... | Do this |
|---|---|
| Add a tool | Subclass `ToolBase`, pass to `FunctionCallEnvironment(tools=[...])` |
| Add a reward | Implement `ResultRewardStrategy` or `ProcessRewardStrategy`, plug into `RewardPipeline` |
| Support a new model | Subclass `ChatProtocol`, implement `render_messages` + `parse_assistant_text` |
| Add a backend | Implement `LLMEngine` (`generate`, `chat`, `tokenize`) |
| Enable compact | Use `CompactableSession` instead of `AgentSession`, set `max_context_tokens` on runtime |
