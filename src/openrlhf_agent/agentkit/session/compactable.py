"""CompactableSession — AgentSession with compaction support."""

from __future__ import annotations

from typing import Optional

from .base import AgentSession


COMPACT_USER_PROMPT = """
Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. Ensure that this step is DIRECTLY in line with the user's most recent explicit requests. If your last task was concluded, then only list next steps if they are explicitly in line with the users request.

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
