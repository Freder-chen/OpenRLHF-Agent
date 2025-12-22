"""Shared Gradio UI helpers for streaming AgentRuntime demos."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from openrlhf_agent.agentkit.runtime import AgentRuntime


def format_step(step: Dict[str, Any]) -> str:
    """Render one message from run_steps into readable text with collapsible sections."""

    def make_section(title: str, body: str, open_default: bool = True) -> str:
        """生成一个折叠块：<details><summary>Title</summary>body</details>"""
        if not body:
            return ""
        open_attr = " open" if open_default else ""
        return (
            f"<details{open_attr}>"
            f"<summary><strong>{title}</strong></summary>\n\n"
            f"{body}\n\n"
            f"</details>"
        )

    role = step.get("role")
    if role == "assistant":
        content = step.get("content", "") or ""
        reasoning = step.get("reasoning_content", "") or ""
        tool_calls = step.get("tool_calls") or []

        parts: List[str] = []

        # Thinking：默认展开
        thinking_block = make_section("Thinking", reasoning, open_default=True)
        if thinking_block:
            parts.append(thinking_block)

        # Tool Calls：默认收起
        if tool_calls:
            tool_body = ", ".join(
                call.get("name", "") for call in tool_calls if isinstance(call, dict)
            )
            if tool_body:
                tool_calls_block = make_section("Tool Calls", tool_body, open_default=False)
                parts.append(tool_calls_block)

        # Final：默认展开
        final_block = make_section("Final", content, open_default=True)
        if final_block:
            parts.append(final_block)

        # 把所有 section 按顺序拼起来
        return "\n\n".join(parts).strip()

    if role == "tool":
        payload = step.get("content", "") or ""
        # Tool Result 也做成可折叠，默认收起（你可以根据需要改成 True）
        return (
            "<details>"
            "<summary><strong>Tool Result</strong></summary>\n\n"
            f"{payload}\n\n"
            "</details>"
        )

    # 其他角色直接原样输出
    return step.get("content", "") or ""




# 终端启动 NO UI
async def run_console(runtime: AgentRuntime, user_question: str) -> None:
    """Minimal console runner that prints streaming steps."""

    messages = [{"role": "user", "content": user_question}]
    async for step in runtime.run_steps(messages):
        print(step)
        print("-" * 100)


def launch_runtime_ui(
    *,
    runtime_builder: Callable[[], AgentRuntime],
    title: str,
    description: str,
    port: int,
    share: bool,
    open_browser: bool = False,
    default_question: Optional[str] = None,
) -> None:
    """Launch a ChatGPT-like Gradio chat UI for an AgentRuntime."""

    async def chat_fn(message: str, history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
        if not message or not message.strip():
            yield history, message
            return

        rt = runtime_builder()
        updated_history: List[Dict[str, str]] = list(history or [])
        updated_history.extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ""},
            ]
        )
        assistant_idx = len(updated_history) - 1
        assistant_buffer = ""

        yield updated_history, ""

        async for step in rt.run_steps([{"role": "user", "content": message}]):
            chunk = format_step(step)
            if not chunk:
                continue
            assistant_buffer = f"{assistant_buffer}\n\n{chunk}".strip() if assistant_buffer else chunk
            updated_history[assistant_idx]["content"] = assistant_buffer
            yield updated_history, ""

        if not assistant_buffer:
            updated_history[assistant_idx]["content"] = "(no response)"
            yield updated_history, ""

    def clear_fn() -> Tuple[List[Dict[str, str]], str]:
        return [], (default_question or "")

    css_text: Optional[str] = None
    for name in ("custom.css", "custom_css.css"):
        css_path = Path(__file__).resolve().parent / name
        if not css_path.exists():
            continue
        try:
            css_text = css_path.read_text(encoding="utf-8")
            break
        except Exception:
            css_text = None

    with gr.Blocks(title="Search-R1 Chat") as demo:
        with gr.Column(elem_id="layout"):
            gr.Markdown(title, elem_id="page-title")
            if description:
                gr.Markdown(description, elem_id="page-subtitle")
            
            chatbot = gr.Chatbot(
                value=[],
                height=500,
                elem_id="chat-area",
                show_label=False,
                render_markdown=True,
                buttons=None,
            )
            msg = gr.Textbox(
                show_label=False,
                placeholder="输入你的问题，按 Enter 发送 / Ask anything...",
                value=default_question or "",
                lines=1,
                max_lines=8,
                elem_id="message-box",
                autofocus=True,
                scale=20,
                container=False,
            )
            with gr.Row(elem_id="input-button", equal_height=True):
                submit = gr.Button("Submit", variant="primary", elem_id="send-btn", scale=1)
                clear = gr.Button("Reset", variant="secondary", elem_id="reset-btn", scale=1)

            gr.Markdown("工具：本地搜索 + 评论。思考链和工具调用会以分段显示。", elem_id="helper-text")

        submit.click(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])
        msg.submit(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])
        clear.click(clear_fn, outputs=[chatbot, msg])

    demo.queue() # 队列模式，支持多用户
    server_port = port if port and port > 0 else None  # None lets Gradio pick a free port
    demo.launch(
        server_port=server_port,
        share=share,
        inbrowser=open_browser,
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="gray", neutral_hue="gray"),
        css=css_text,
    )
