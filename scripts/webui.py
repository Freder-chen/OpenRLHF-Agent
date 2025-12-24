# Local web UI entrypoint.
# python scripts/webui.py

from datetime import datetime
from typing import Any, Dict, List, Optional

import gradio as gr

from openrlhf_agent.backends import OpenAIEngine
from openrlhf_agent.agentkit.factory import build_environment, build_protocol
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.tools import CommentaryTool, LocalSearchTool


SYSTEM_PROMPT_TEMPLATE = """
Answer the given question. First, think step by step inside <think> and </think> whenever you receive new information. 
After reasoning, decide whether to use tools. Use tools to verify specific aspects of your reasoning or to fetch missing knowledge; 
do not rely on tools to write the final answer. Call the commentary tool only for brief progress updates.

If the conditions for solving the problem have been met, directly provide the final answer inside <final> and </final> without extra illustrations. 
Example: <final> ... </final>.

Knowledge cutoff: 2023-06
Current date: {date}
""".strip()


def render_html(messages: List[Dict[str, Any]]) -> str:
    """Render one run_steps item into readable HTML with collapsible sections."""

    detail_sections: List[str] = []
    for message in messages:
        role = message.get("role")

        if role == "tool":
            tool_output = message.get("content") or ""
            detail_sections.append(
                "<details>\n"
                "<summary><strong>Tool Result</strong></summary>\n"
                f"{tool_output}\n"
                "</details>\n\n"
            )

        if role == "assistant":
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            tool_calls = message.get("tool_calls") or []

            if reasoning:
                detail_sections.append(
                    "<details>\n"
                    "<summary><strong>Thinking</strong></summary>\n"
                    f"{reasoning}\n"
                    "</details>\n\n"
                )
            if tool_calls:
                tool_calls_text = "\n".join(str(call) for call in tool_calls)
                detail_sections.append(
                    "<details open>\n"
                    "<summary><strong>Tool Calls</strong></summary>\n"
                    f"{tool_calls_text}\n"
                    "</details>\n\n"
                )
            if content:
                detail_sections.append(
                    f"{content}\n"
                )

    return "".join(section for section in detail_sections if section).strip()


def launch_runtime_ui(
    runtime: AgentRuntime,
    page_title: str,
    page_description: str,
    default_prompt: Optional[str] = None,
) -> gr.Blocks:
    """Build a ChatGPT-like Gradio chat UI for an AgentRuntime."""

    history_messages = []

    async def handle_chat_message(show_messages: List[Dict[str, str]], prompt: str):
        if not prompt.strip():
            return

        history_messages.append({"role": "user", "content": prompt})

        show_messages = history_messages
        yield show_messages, ""

        new_assistant_messages = []
        async for step in runtime.run_steps(history_messages):
            new_assistant_messages.append(step)

            new_show_messages = [{"role": "assistant", "content": render_html(new_assistant_messages) + "\n\n..."}]
            yield show_messages + new_show_messages, ""
        
        new_show_messages = [{"role": "assistant", "content": render_html(new_assistant_messages)}]
        yield show_messages + new_show_messages, ""

        assert new_assistant_messages[-1]["role"] == "assistant"
        history_messages.append(new_assistant_messages[-1])

    with gr.Blocks(title="OpenRLHF-Agent") as ui:
        with gr.Column(elem_id="layout"):
            gr.Markdown(page_title, elem_id="page-title")
            if page_description:
                gr.Markdown(page_description, elem_id="page-subtitle")

            gr.Markdown(f"Available Tools: {runtime.session.environment.tool_names()}", elem_id="helper-text")

            chat_panel = gr.Chatbot(
                value=[],
                height=500,
                elem_id="chat-area",
                show_label=False,
                render_markdown=True,
                buttons=None,
            )
            message_box = gr.Textbox(
                show_label=False,
                placeholder="Enter your question and press Enter / Ask anything...",
                value=default_prompt or "",
                lines=1,
                max_lines=8,
                elem_id="message-box",
                autofocus=True,
                scale=20,
                container=False,
            )
            with gr.Row(elem_id="input-button", equal_height=True):
                send_button = gr.Button("Submit", variant="primary", elem_id="send-btn", scale=1)
                reset_button = gr.Button("Reset", variant="secondary", elem_id="reset-btn", scale=1)

        send_button.click(handle_chat_message, inputs=[chat_panel, message_box], outputs=[chat_panel, message_box])
        message_box.submit(handle_chat_message, inputs=[chat_panel, message_box], outputs=[chat_panel, message_box])
        reset_button.click(lambda: ([], default_prompt or ""), outputs=[chat_panel, message_box])

    return ui


if __name__ == "__main__":
    # Configure the agent runtime (engine, environment, protocol).
    agent_runtime = AgentRuntime(
        engine=OpenAIEngine(
            model="qwen3",
            base_url="http://localhost:8009/v1",
            api_key="empty",
        ),
        environment=build_environment(
            name="function_call",
            tools=[
                CommentaryTool(),
                # LocalSearchTool(base_url="http://localhost:8000/retrieve"),
            ],  # Tools used by the runtime and UI.
            system_prompt=SYSTEM_PROMPT_TEMPLATE.format(date=datetime.now().strftime("%Y-%m-%d")),
        ),
        protocol=build_protocol(name="qwen3_thinking"),
    )

    # Build the UI and start the server.
    ui = launch_runtime_ui(
        runtime=agent_runtime,
        page_title="## OpenRLHF-Agent",
        page_description=None,
        default_prompt="Please use the commentary function to share your thoughts, and also help me search what Python is?",
    )
    ui.launch(
        server_port=7867,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="gray", neutral_hue="gray"),
        css="body, * { font-family: \"Times New Roman\", SimSun, sans-serif; }",
    )
