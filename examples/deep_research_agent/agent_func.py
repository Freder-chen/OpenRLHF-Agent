from datetime import datetime
import torch
from typing import Any, Dict

from openrlhf_agent.agentkit.rewards import RewardPipeline
from openrlhf_agent.agentkit.session import AgentSession
from openrlhf_agent.agentkit.environments import FunctionCallEnvironment
from openrlhf_agent.agentkit.protocols import Qwen3ThinkingProtocol
from openrlhf_agent.agentkit.rewards.result_rewards import MathMatchingReward
# from openrlhf_agent.agentkit.tools import CommentaryTool, LocalSearchTool

from deep_research_agent.reward.report_reward import ReportRewardStrategy
from deep_research_agent.tool.my_tool_test_time_v2 import BaiduSearchTool, Crawler360Tool, JinaReaderTool

from openrlhf.utils.agent import MultiTurnAgentExecutor, AgentInstanceBase


CUSTOM_SYSTEM_PROMPT = """
你是一名“首席情报整合官”。你的核心职责是通过调用互联网搜索功能及多轮推理，生成一份面向政府或媒体深度研究的并**可直接交付 MarkDown 报告**。整个过程中，必须全程由你来完整规划，不要与我交互。另外一定不要展示思考过程与闲聊，不要使用表格。

    ### 你的工作流程（严格执行）：

    **第一阶段：全域情报采集（必须调用以下三个工具，缺一不可）**
    1. 调用 `BaiduSearchTool`： 面向互联网的关键词检索，用于快速收集相关内容。
    2. 调用 `Crawler360Tool`：网页正文内容抓取。
    3. 调用 `JinaReaderTool`：网页正文内容抓取。

    **第二阶段：构建最终 MarkDown 报告**
    1. 报告必须为 **完整的 MarkDown 页面**。 
    2. 正文引用必须采用 **参考的表述信息[数字]** 的形式，例如："李强提出社会阶层划分模型[15]"。 
    3. 所有参考文献必须在文末统一列出，格式必须采用 **[数字] URL** 的形式,我将用于溯源,例如： [1] https://example.com/article123 [2] https://another-source.org/report456 
    4. 文末参考文献的序号必须与正文引用保持一致，且按升序排列。 
    5. 不允许在正文中直接贴原始链接，必须通过序号引用。 
    6. 报告长度不少于 20000 字，避免重复。
Current date: {date}
""".strip()


class AgentInstance(AgentInstanceBase):
    def __init__(self, *args, **kwargs):
        environment = FunctionCallEnvironment(
            system_prompt=CUSTOM_SYSTEM_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d")),
            tools=[
                BaiduSearchTool(),
                Crawler360Tool(),
                JinaReaderTool()
            ],
        )
        protocol = Qwen3ThinkingProtocol()
        pipeline = RewardPipeline(
            result_reward=ReportRewardStrategy(miss_score=0.0),
        )
        self.session = AgentSession(environment=environment, protocol=protocol, reward_pipeline=pipeline)

    async def reset(self, states: dict, **kwargs):
        prompt = await self.session.initialize(states.get("observation"))
        return {"observation": prompt}

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        action_text: str = states.get("action_text", "")
        label = states.get("label")

        observation, reward = await self.session.step_from_text(action_text, label=label)

        reward = float(reward) if reward is not None else 0.0
        reward = max(reward, -1.0)

        done = observation.done
        return {
            "rewards": torch.tensor(reward),
            "scores": torch.tensor(reward),
            "environment_feedback": "" if done else observation.feedback_text,
            "done": done,
            "sampling_params": states.get("sampling_params", None),
            "extra_logs": {
                "dummy_scores": torch.tensor(reward),
                "turn_count": torch.tensor(observation.step_index),
            },
        }


class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(AgentInstance)
