"""CompactableSession — AgentSession with compaction support."""

from __future__ import annotations

from typing import Optional

from .base import AgentSession


COMPACT_USER_PROMPT = """
<harness_call>
Context compression requested by the agent runtime. This is not a user message.
Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail.
2. Key Findings: List the important facts, data, and conclusions discovered so far. Be specific — include names, numbers, URLs, and other concrete details.
3. Tool Usage: For each tool call, document what was called, with what inputs, and what key results were returned. Pay special attention to the most recent tool calls.
4. Errors and Corrections: List any errors encountered and how they were resolved. Include specific user feedback if they told you to do something differently.
5. All User Messages: List ALL user messages that are not tool results. These are critical for understanding the user's feedback and changing intent.
6. Pending Tasks: Outline any tasks you have been explicitly asked to work on but have not completed.
7. Current Work: Describe precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages.
8. Next Step: List the next step directly in line with the user's most recent request.

Please provide your summary directly, following this structure. Ensure precision and thoroughness — this summary will replace the full conversation history.
</harness_call>
""".strip()

RESUME_PROMPT_TEMPLATE = """
<harness_call>
This session is being continued from a previous conversation that ran out of context.
The summary below covers what has been done so far.

{summary}

Based on the summary above, continue working on the pending tasks. Do not re-introduce yourself or ask what to do — pick up where you left off.
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
