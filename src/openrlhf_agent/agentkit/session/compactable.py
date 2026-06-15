"""CompactableSession — AgentSession with compaction support."""

from __future__ import annotations

from typing import Optional

from .base import AgentSession


COMPACT_USER_PROMPT = """
Context compression requested by the agent runtime. This is not a user message.
Summarize the conversation above so the agent can continue working from the summary.

Your summary should include the following sections:
1. User's requests and intents
2. Key findings and conclusions
3. Tool calls: what was called, inputs, key results
4. Errors encountered and how they were resolved
5. All user messages (not tool results) — tracks changing intent
6. Pending tasks not yet completed
7. What was being worked on immediately before this request
8. Next step (only if explicitly requested by the user)

Please provide your summary directly, following this structure.
""".strip()

RESUME_PREFIX = (
    "This session is being continued from a previous conversation that ran out "
    "of context. The summary below covers the earlier portion of the conversation."
)


class CompactableSession(AgentSession):
    """AgentSession that can compact. Caller decides when to trigger.

    Two-step compact:
    1. ``request_compact()`` — returns feedback_text (compact instruction
       rendered as a user turn). Caller appends to prompt_ids and generates.
    2. ``finish_compact(summary)`` — re-initializes with the summary.
       Returns new prompt text.
    """

    def request_compact(self, focus: Optional[str] = None) -> str:
        """Return compact instruction as feedback_text.

        The caller tokenizes this and appends to prompt_ids,
        then lets the model generate a summary.
        """
        instruction = COMPACT_USER_PROMPT
        if focus:
            instruction += f"\n\nAdditional summarization instructions:\n{focus}"

        return self.protocol.render_messages(
            messages=[{"role": "user", "content": instruction}],
            add_generation_prompt=True,
        )

    async def finish_compact(self, summary: str) -> str:
        """Re-initialize the session with the generated summary.

        Returns the new prompt text.
        """
        payload = [
            {"role": "user", "content": f"{RESUME_PREFIX}\n\n{summary}"},
        ]
        return await self.initialize(payload)
