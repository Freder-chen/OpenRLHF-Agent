"""CompactableSession — AgentSession with compaction support."""

from __future__ import annotations

from typing import Optional

from .base import AgentSession


COMPACT_USER_PROMPT = """
<harness_call>
Context compression requested by the agent runtime.
Summarize the conversation above so the agent can continue working from the summary.

Your summary should include the following sections:
1. User's requests and intents
2. Key findings and conclusions
3. Tool calls: what was called, inputs, key results
4. Errors encountered and how they were resolved
5. All user messages (not tool results) — tracks changing intent
6. Pending tasks not yet completed
7. What was being worked on immediately before this request
8. Next step

Please provide your summary directly, following this structure.
</harness_call>
""".strip()

RESUME_PROMPT_TEMPLATE = """
<harness_call>
This session is being continued from a previous conversation that ran out of context.
The summary below covers what has been done so far.

{summary}

Continue working on the user's request based on this summary.
</harness_call>
""".strip()


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
        return await self.initialize([
            {"role": "user", "content": RESUME_PROMPT_TEMPLATE.format(summary=summary)},
        ])
