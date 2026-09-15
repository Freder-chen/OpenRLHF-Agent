from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional
import json
import logging
from jinja2 import Environment
import re 
import asyncio

from openrlhf_agent.utils.types import Action, RewardSample
from openrlhf_agent.agentkit.rewards.result_rewards import ResultRewardStrategy
from deep_research_agent.reward.api import AIClient
from deep_research_agent.prompt.clean_prompt import clean_article_system_prompt_zh, clean_article_user_prompt_zh
from deep_research_agent.prompt.score_prompt_zh import generate_dynamic_score_system_prompt_zh_no_ref, generate_dynamic_score_user_prompt_zh_no_ref

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

_PROMPT_ENV = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
_QUESTION_TEMPLATE = _PROMPT_ENV.from_string(
    """
    {% for message in messages -%}
    [{{ message.role|capitalize }}]
    {{ message.content or "" }}
    {% if message.tool_calls %}
    {% for tool_call in message.tool_calls -%}
    {{ tool_call }}
    {% if not loop.last %}

    {% endif %}
    {% endfor %}
    {% endif %}
    {% if not loop.last %}

    {% endif %}
    {% endfor %}
    """.strip()
)

def _render_tool_call_payload(payload: Any) -> Optional[str]:
    """Return a compact, human-readable representation of a tool call."""

    if payload is None:
        return None

    if isinstance(payload, Mapping):
        data = dict(payload)
    elif hasattr(payload, "model_dump"):
        data = payload.model_dump(exclude_none=True)
    else:
        return None

    call_id = data.get("call_id") or data.get("id")
    arguments: Any = data.get("arguments")
    name = data.get("name")

    if not name and isinstance(data.get("function"), Mapping):
        function_payload = data["function"]
        name = function_payload.get("name", name)
        if arguments is None:
            arguments = function_payload.get("arguments")

    if isinstance(arguments, str):
        args_text = arguments
    elif arguments is None:
        args_text = "{}"
    else:
        try:
            args_text = json.dumps(arguments, ensure_ascii=False)
        except TypeError:
            args_text = str(arguments)

    lines = ["<tool_call>"]
    if call_id:
        lines.append(f"id: {call_id}")
    if name:
        lines.append(f"name: {name}")
    lines.append(f"arguments: {args_text}")
    lines.append("</tool_call>")
    return "\n".join(lines)

def _normalize_messages(payload: Iterable[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for entry in payload:
        if entry is None:
            continue

        if isinstance(entry, Mapping):
            data = dict(entry)
        elif hasattr(entry, "model_dump"):
            data = entry.model_dump(exclude_none=True)
        else:
            continue

        normalized.append(
            {
                "role": data.get("role"),
                "content": data.get("content"),
                "tool_calls": [
                    rendered for rendered in (
                        _render_tool_call_payload(call)
                        for call in data.get("tool_calls") or []
                    ) if rendered
                ] or None,
            }
        )
    return normalized

# 获取 query
def render_question_from_sample(sample: Optional[RewardSample]) -> str:
    if not sample or not sample.question:
        return ""

    question_payload = sample.question
    if isinstance(question_payload, str):
        return question_payload.strip()

    formatted_messages = _normalize_messages(question_payload)
    if not formatted_messages:
        return ""

    return _QUESTION_TEMPLATE.render(messages=formatted_messages).strip()

# 暂时未用
def _get_language(data: dict, field: str = "language") -> str:
    # 假设从 data field 字段读出语言类别
    language = data.get(field, None)
    if language is None:
        return "zh"
    return "en" if str(language).lower() == "en" else "zh"

def extract_json_from_markdown(text):
    if not isinstance(text, str):
        return None
    
    # --- 健壮性增强 1: 预处理文本 ---
    # 移除可能存在的控制字符（这些字符常导致 json.loads 失败）
    # 同时去掉开头结尾的空白
    text = "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\r\t").strip()

    # --- 阶段 A: 标准/结构化提取 (Method 0-5 整合优化) ---
    
    # 尝试寻找 Markdown 代码块
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block_match:
        potential_jsons = [code_block_match.group(1).strip()]
    else:
        # 如果没有代码块，尝试提取最外层的 {} 或 []
        potential_jsons = []
        # 匹配嵌套的大括号
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            potential_jsons.append(text[start_idx:end_idx+1])
        # 匹配方括号
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx != -1 and end_idx != -1:
            potential_jsons.append(text[start_idx:end_idx+1])

    for pj in potential_jsons:
        # 尝试清理常见小错误
        cleaned_pj = pj.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        try:
            json.loads(cleaned_pj)
            return cleaned_pj
        except json.JSONDecodeError:
            continue

    # --- 阶段 B: 强力正则兜底 (针对新格式 Method 6 修改) ---
    # 即使 LLM 输出了非法的 {{ [...] }} 也能救回来
    if "target_score" in text and "criterion" in text:
        try:
            # 1. 提取所有字段 (考虑了引号可能被模型写错或缺失的情况)
            criteria = re.findall(r'"criterion"\s*:\s*"([\s\S]*?)"(?=[,\s\}])', text)
            analyses = re.findall(r'"analysis"\s*:\s*"([\s\S]*?)"(?=[,\s\}])', text)
            scores = re.findall(r'"target_score"\s*:\s*(\d+\.?\d*)', text)

            # 2. 组装结果
            result_list = []
            # 按照最小公共长度对齐
            for i in range(min(len(criteria), len(analyses), len(scores))):
                result_list.append({
                    "criterion": criteria[i].strip(),
                    "analysis": analyses[i].strip(),
                    "target_score": float(scores[i])
                })
            
            if result_list:
                return json.dumps(result_list, ensure_ascii=False)
        except Exception:
            pass

    return None

def calculate_weighted_scores(judge_score, rubrics):
    if not judge_score or not rubrics:
        print('Judge 为空或 rubrics 为空')
        return 0.0

    rubric_map = {r['description'].strip().lower(): r for r in rubrics}
    
    total_weighted_points = 0.0
    total_weight_sum = sum(r.get('weight', 0) for r in rubrics)
    
    
    if total_weight_sum == 0:
        print(f'rubrics 权重加和为零{rubrics}')
        return 0.0

    matched_weight = 0.0

    for item in judge_score:
        criterion_text = item.get('criterion', '').strip().lower()
        title_text = item.get('title', '').strip().lower()
        score = item.get('target_score', 0)
        
        target_rubric = rubric_map.get(criterion_text) or rubric_map.get(title_text)
        if target_rubric is not None:
            weight = target_rubric.get('weight', 0)
        else:
            print(f"Judge 与 rubrics 不匹配 {criterion_text}")
            weight = 0
            
        total_weighted_points += score * weight
        matched_weight += weight

    final_score = total_weighted_points / total_weight_sum
    
    return final_score

def reward_power(reward, p=3):
    return (reward / 10) ** p

@dataclass
class ReportRewardStrategy(ResultRewardStrategy):
    miss_score: float = 0.0

    def __post_init__(self):
        self.llm_client = AIClient()
    
    # 最小清洗文章单元
    async def _clean(self, text: str, max_retries=3):
        clearn_system_prompt, clean_user_prompt = clean_article_system_prompt_zh, clean_article_user_prompt_zh
        clean_user_prompt = clean_user_prompt.format(article=text)
        
        for retry in range(max_retries):
            try:
                result = await self.llm_client.generate(user_prompt=clean_user_prompt, system_prompt=clearn_system_prompt)
                if result is not None:
                    return result
                logger.warning(f"cleaning result is none, retry #{retry+1}")
            except Exception as e:
                logger.error(f"llm_client generate call error: {e}")
        
        return None

    # 分块清洗文章
    async def _clean_article_chunk(self, text: str):
        logger.info("Attempting to process article in 2 chunks concurrently")
        
        chunks = []
        chunk_size = len(text) // 2
        
        # 1. 依然保持你原有的语义分块逻辑
        for i in range(2):
            start = i * chunk_size
            end = len(text) if i == 1 else chunk_size
            
            if i == 0:
                search_start = max(0, end - 200)
                for j in range(end, search_start, -1):
                    if j < len(text) and text[j] in ['.', '?', '!', '。', '？', '！', '\n']:
                        end = j + 1
                        break
            
            chunks.append(text[start:end])
        
        # 2. 核心修改：创建任务列表，不立即 await
        tasks = [self._clean(chunk) for chunk in chunks]
        
        try:
            # 3. 并发执行所有任务
            logger.info(f"Dispatching {len(tasks)} cleaning tasks...")
            cleaned_chunks = await asyncio.gather(*tasks)
            
            # 4. 结果校验：只要有一个块清洗失败（None），整体就返回 None
            if any(res is None for res in cleaned_chunks):
                logger.error("One or more chunks failed to clean.")
                return None
            
            logger.info("All chunks processed concurrently, merging results")
            return "".join(cleaned_chunks)
            
        except Exception as e:
            logger.error(f"Concurrent cleaning failed: {e}")
            return None

    async def clean_text(self, text: str, max_retries=3):
        result = await self._clean(text, max_retries)

        if result is not None:
            return result

        result = await self._clean_article_chunk(text)
        return result

    async def generate_score(self, question: str, clean_response: str, rubrics: str, max_retries=3):
        system_prompt, user_prompt = generate_dynamic_score_system_prompt_zh_no_ref, generate_dynamic_score_user_prompt_zh_no_ref
        user_prompt = user_prompt.format(task_prompt=question, article=clean_response, criteria_list=rubrics)
        llm_response_str = ""
        for _ in range(max_retries):
            try:
                llm_response_str = await self.llm_client.generate(user_prompt=user_prompt, system_prompt=system_prompt)
                if llm_response_str is not None:
                    break
            except Exception as e:
                continue
 
        if llm_response_str is None:
            logger.error(f"Judge 打分失败")
            return None
        
        try:
            json_str_extracted = extract_json_from_markdown(llm_response_str)
            if json_str_extracted is None:
                raise ValueError("无法从 Judge 中提取有效内容")
        except Exception as e:
            logger.error(f"{e}")
            return None
        
        return json_str_extracted
            
    def cal_reward(self, json_str_extracted: str, rubrics: str):
        try: 

            judge_score = json.loads(json_str_extracted)
            rubrics = json.loads(rubrics)
            scores = calculate_weighted_scores(judge_score, rubrics)
            return scores
        except Exception as e:
            logger.error(f"There are errors in calculating the reward: {e}")
            return 0.0

    async def score_response(self, question: str, response: str, rubrics: str) -> float:
        try:
            # clean_response = self.clean_text(response)
            clean_response = response

            if clean_response is None:
                return 0.0
            
            json_str_extracted = await self.generate_score(question, clean_response, rubrics)
            reward = self.cal_reward(json_str_extracted, rubrics)
            reward = reward_power(reward)
            return reward
        except Exception as e:
            return 0.0

    async def score(
        self,
        *,
        action: Action,
        label: Optional[Any],
        sample: Optional[RewardSample] = None,
    ) -> float:
        response = self.extract_final_response(action)
        question_text = render_question_from_sample(sample)
        rubrics = label
        # 若缺少 问题、回答、评分标准 任一种返回零
        if response is None or question_text is None or rubrics is None:
            return self.miss_score

        return await self.score_response(question_text, response, rubrics)


if __name__ == '__main__':
    example_question = 'do a literature review on cable anchor and also their potential to replace timber support in hardrock mines'
    example_label = """
    [
        {
            "description": "Define cable anchors, their basic purpose, and their core mechanical principles in ground support.",
            "title": "Describe cable anchors and core mechanics",
            "weight": 3
        },
        {
            "description": "Explain the use of cable anchors/bolts in hardrock mines, including current deployment, prevalence, and examples.",
            "title": "Discuss cable anchor use in hardrock mining",
            "weight": 3
        },
        {
            "description": "Compare the mechanical properties and support capabilities of cable anchors/bolts versus timber supports in hardrock environments.",
            "title": "Compare cable anchors to timber supports",
            "weight": 3
        },
        {
            "description": "Evaluate the potential for cable anchors to replace timber supports in hardrock mines, specifically addressing performance, adaptability, and current research directions.",
            "title": "Assess replacement of timber supports by cable anchors",
            "weight": 3
        }
    ]
    """

    example_answer = """
        在基础定义与力学原理方面，被评估文章表现得非常专业。它不仅准确界定了电缆锚杆作为深部加固元件的角色，还详细阐述了预应力钢绞线与围岩之间的载荷传递机制。文章通过对比“悬吊效应”与“挤压加固效应”，使读者能够清晰理解其力学本质。基于此，在该项标准上，文章展现了扎实的理论基础，分析逻辑严密，几乎没有瑕疵。
        针对硬岩矿山应用现状的讨论，文章提供了全球范围内主流硬岩矿山的部署实例，涵盖了铜矿、金矿等不同矿种。文中提到的自动化锚杆台车（Cable Bolter）的普及，有效论证了该技术在现代化矿山中的主导地位。不过，如果能进一步增加关于不同岩石质量指数（RMR）下锚杆排列间距的量化数据，评估内容将更加完美。总体而言，这部分内容详实，符合大部分评估要求。
        在与木支护的对比分析标准下，文章的优点在于多维度的直观对比。作者从主动支护与被动支护的区别入手，深入分析了木支护在深部高应力环境下易发生脆性失效的弱点。相比之下，电缆锚杆在提供支护强度的同时，不占用巷道空间的优势被重点强调。分析过程结合了力学性能曲线，证据充分，极具说服力。
        最后，关于替代潜力评估，文章的前瞻性值得称赞。它不仅讨论了技术上的可行性，还分析了在地震活动频繁的硬岩矿山中，吸能型（Yielding）电缆锚杆对木支护的绝对优势。虽然文中简要提到了初始投资成本的问题，但对长期经济效益的核算略显简单。尽管如此，文章对未来研究方向（如智能传感锚杆）的捕捉非常敏锐。
    """
    fake_action = Action(
        content=example_answer
    )
    fake_sample = RewardSample(
        question=example_question
    )
    fake_label = example_label

    report_reward = ReportRewardStrategy()
    reward = asyncio.run(report_reward.score(action=fake_action, label=fake_label, sample=fake_sample) )
    print(reward)
