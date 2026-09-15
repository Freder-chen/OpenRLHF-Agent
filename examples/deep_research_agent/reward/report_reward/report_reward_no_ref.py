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
from deep_research_agent.prompt.clean_prompt import clean_article_system_prompt_zh, clean_article_user_prompt_zh, clean_article_system_prompt_en, clean_article_user_prompt_en
from deep_research_agent.prompt.score_prompt_zh import generate_static_score_system_prompt_zh_no_ref, generate_static_score_user_prompt_zh_no_ref, point_wise_score_prompt
# from openrlhf_agent.agentkit.rewards.rubric import static_rubric
# from openrlhf_agent.agentkit.rewards.example import article

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
    """Extract JSON from a markdown text that may contain ```json ... ``` blocks
    
    Args:
        text (str): The input text which might contain JSON blocks
        
    Returns:
        str or None: The extracted JSON string or None if not found/not valid
    """
    if not isinstance(text, str):
        return None
    
    # Method 0: Try to parse the complete text directly, if it's valid JSON return it
    # This handles cases where LLM returns a direct JSON object
    if text.strip().startswith('{') and text.strip().endswith('}'):
        try:
            # Only validate, no need to assign to variable
            json.loads(text.strip())
            return text.strip()
        except json.JSONDecodeError:
            # If not valid JSON, continue with other methods
            pass
    
    # Method 1: Use string operations to extract JSON directly from code blocks
    if "```json" in text and "```" in text[text.find("```json")+7:]:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            json_str = text[start:end].strip()
            try:
                # Validate JSON
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                # If validation fails, continue with other methods
                pass
    
    # Method 2: As backup, retain original regex matching
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        json_str = match.group(1).strip()
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            # If validation fails, continue with other methods
            pass
    
    # Method 3: Check if the entire text is directly JSON
    try:
        json_obj = json.loads(text.strip())
        return text.strip()
    except json.JSONDecodeError:
        # Not valid JSON, continue trying
        pass
    
    # Method 4: Try to extract content within outermost curly braces
    # Find the first left curly brace
    start = text.find('{')
    if start != -1:
        # Calculate nesting level to find the matching closing brace
        level = 0
        for i, char in enumerate(text[start:]):
            if char == '{':
                level += 1
            elif char == '}':
                level -= 1
                if level == 0:
                    # Found matching closing brace
                    end = start + i + 1
                    potential_json = text[start:end]
                    try:
                        json.loads(potential_json)
                        return potential_json
                    except json.JSONDecodeError:
                        # If validation fails, continue with other methods
                        pass
                    break
    
    # Method 5: Final fallback method, try simple start and end brace matching
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        potential_json = text[start:end+1]
        try:
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            # All methods failed, try final fallback method
            pass
    
    # Method 6: Special fallback method - build simplified JSON object by keyword pattern matching
    if "comprehensiveness" in text and "article_1_score" in text and "article_2_score" in text:
        try:
            dimensions = ["comprehensiveness", "insight", "instruction_following", "readability"]
            result = {}
            
            for dim in dimensions:
                if dim in text:
                    result[dim] = []
                    
                    # Search for all scoring entries under this dimension
                    # First find the dimension starting position
                    dim_start = text.find(f'"{dim}"') 
                    if dim_start == -1:
                        dim_start = text.find(f"'{dim}'")
                    if dim_start == -1:
                        dim_start = text.find(dim)
                    
                    if dim_start != -1:
                        # Determine dimension end position (next dimension start or end of text)
                        next_dim_start = len(text)
                        for next_dim in dimensions:
                            if next_dim != dim:
                                pos = text.find(f'"{next_dim}"', dim_start)
                                if pos == -1:
                                    pos = text.find(f"'{next_dim}'", dim_start)
                                if pos == -1:
                                    pos = text.find(next_dim, dim_start + len(dim))
                                if pos != -1 and pos < next_dim_start:
                                    next_dim_start = pos
                        
                        # Extract content for this dimension
                        dim_content = text[dim_start:next_dim_start]
                        
                        # Use regex to find all "criterion", "article_1_score" and "article_2_score"
                        criterion_matches = re.finditer(r'"criterion"\s*:\s*"([^"]+)"', dim_content)
                        score1_matches = re.finditer(r'"article_1_score"\s*:\s*(\d+\.?\d*)', dim_content)
                        score2_matches = re.finditer(r'"article_2_score"\s*:\s*(\d+\.?\d*)', dim_content)
                        
                        # Convert to lists for multiple access
                        criteria = [m.group(1) for m in criterion_matches]
                        scores1 = [float(m.group(1)) for m in score1_matches]
                        scores2 = [float(m.group(1)) for m in score2_matches]
                        
                        # Combine into scoring entries
                        for i in range(min(len(criteria), len(scores1), len(scores2))):
                            result[dim].append({
                                "criterion": criteria[i],
                                "article_1_score": scores1[i],
                                "article_2_score": scores2[i]
                            })
            
            # Validate if we successfully extracted scoring data
            if any(len(scores) > 0 for scores in result.values()):
                return json.dumps(result)
        except Exception as e:
            # Fallback extraction method failed, log error but continue trying
            pass
    
    # All methods failed, return None
    return None 

def calculate_weighted_scores(llm_output_json, criteria_data):
    """
    Returns:
        dict:
            {
              "target": {"dims": {f"{dim}_weighted_avg": float}, "total": float},
              "reference": {"dims": {f"{dim}_weighted_avg": float}, "total": float}
            }
    """
    llm_output_json = {k.lower(): v for k, v in llm_output_json.items()}
    results = {
        "target": {"dims": {}, "total": 0.0},
        "reference": {"dims": {}, "total": 0.0}
    }
    total_target_score = 0.0
    total_reference_score = 0.0

    # 获取不同维度权重
    dimension_weights = criteria_data.get("dimension_weight", {})

    # 确保包含详细权重
    if "criterions" not in criteria_data or not criteria_data["criterions"]:
        error_msg = "Missing required criterions data, cannot calculate weighted scores"
        logging.error(error_msg)
        raise ValueError(error_msg)

    criterion_weights = {}
    # 遍历 k v 对
    for dim, criterions in criteria_data.get("criterions", {}).items():
        try:
            criterion_weights[dim] = {crit["criterion"]: float(crit["weight"]) for crit in criterions}
        except (KeyError, TypeError, ValueError):
            logging.warning(f"Invalid criterions format in dimension '{dim}'. Skipping this dimension.")
            criterion_weights[dim] = {}

    unmatched_criteria = set()

    for dim, scores_list in llm_output_json.items():
        if not isinstance(scores_list, list):
            logging.warning(f"Dimension '{dim}' in LLM output is not a list. Skipping.")
            continue

        if dim not in dimension_weights:
            logging.warning(f"Dimension '{dim}' not found in dimension_weight. Skipping dimension.")
            continue

        if dim not in criterion_weights or not criterion_weights[dim]:
            logging.warning(f"Dimension '{dim}' not found in criterions or empty. Skipping dimension.")
            continue

        dim_target_weighted_sum = 0.0
        dim_reference_weighted_sum = 0.0
        dim_total_weight = 0.0

        dim_criteria_map = criterion_weights.get(dim, {})

        # 迭代每一维度大模型打分
        for score_item in scores_list:
            if not isinstance(score_item, dict):
                logging.warning(f"Item in scores_list for dimension '{dim}' is not a dictionary. Skipping item: {score_item}")
                continue

            criterion_text_raw = score_item.get("criterion")
            # 大模型打分对应标准
            criterion_text = criterion_text_raw.strip() if isinstance(criterion_text_raw, str) else None

            article_1_score_raw = score_item.get("article_1_score")
            article_2_score_raw = score_item.get("article_2_score")
            target_score_raw = score_item.get("target_score")  # Single scoring mode

            # 无需参考文档打分
            if target_score_raw is not None and article_1_score_raw is None:
                article_1_score_raw = target_score_raw

            try:
                article_1_score = float(article_1_score_raw) if article_1_score_raw is not None else None
                article_2_score = float(article_2_score_raw) if article_2_score_raw is not None else None
            except (ValueError, TypeError):
                logging.warning(f"Invalid score format for criterion '{criterion_text}' in dimension '{dim}'. Skipping criterion.")
                continue

            if criterion_text and article_1_score is not None:
                # 根据大模型打分标准确定规则对应权重
                weight = dim_criteria_map.get(criterion_text)

                # 未找到权重
                if weight is None:
                    # 大模型打分标准转为小写
                    criterion_lower = criterion_text.lower()
                    # 指定维度规则中迭代寻找权重
                    for key, val in dim_criteria_map.items():
                        if key.lower() == criterion_lower:
                            weight = val
                            break
                    
                    if weight is None:
                        for key, val in dim_criteria_map.items():
                            # 规则与打分标准部分匹配即可
                            if criterion_lower in key.lower() or key.lower() in criterion_lower:
                                weight = val
                                break
                
                # 仍没找到
                if weight is None:
                    unmatched_criteria.add(f"{dim}:{criterion_text}")
                    # 按平均权重处理
                    weight = sum(dim_criteria_map.values()) / len(dim_criteria_map)
                    
                dim_target_weighted_sum += article_1_score * weight
                dim_total_weight += weight
                
                if article_2_score is not None:
                    dim_reference_weighted_sum += article_2_score * weight
            else:
                if criterion_text:
                    if dim not in getattr(calculate_weighted_scores, '_warned_dims', set()):
                        logging.warning(f"Criterion text mismatch for dimension '{dim}': '{criterion_text}'. Available criteria keys start with: {list(dim_criteria_map.keys())[:1] if dim_criteria_map else []}... Check prompt/output/criteria file consistency. Further mismatches in this dim won't be logged fully.")
                        if not hasattr(calculate_weighted_scores, '_warned_dims'): calculate_weighted_scores._warned_dims = set()
                        calculate_weighted_scores._warned_dims.add(dim)
                    else:
                        logging.debug(f"Another criterion mismatch for dimension '{dim}': '{criterion_text}'")
                elif not criterion_text:
                    logging.warning(f"Missing 'criterion' key in score item for dimension '{dim}': {score_item}. Skipping item.")

        if dim_total_weight > 0:
            dim_target_avg = dim_target_weighted_sum / dim_total_weight
            # Only calculate reference average if reference scores exist
            dim_reference_avg = dim_reference_weighted_sum / dim_total_weight if article_2_score is not None else 0
        else:
            dim_target_avg = 0
            dim_reference_avg = 0
            if len(scores_list) > 0:
                logging.warning(f"No valid criteria scored for dimension '{dim}' despite {len(scores_list)} items in LLM output. Check for systematic mismatches. Dimension average set to 0.")

        results["target"]["dims"][f"{dim}_weighted_avg"] = dim_target_avg
        results["reference"]["dims"][f"{dim}_weighted_avg"] = dim_reference_avg

        dim_weight = dimension_weights.get(dim, 0)
        total_target_score += dim_target_avg * dim_weight
        total_reference_score += dim_reference_avg * dim_weight

    # Log warning if there are unmatched criteria
    if unmatched_criteria:
        logging.warning(f"{len(unmatched_criteria)} criteria without exact matches: {unmatched_criteria}")
    
    results["target"]["total"] = total_target_score
    results["reference"]["total"] = total_reference_score

    return results

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

    async def generate_score(self, question: str, clean_response: str, rubrics, max_retries=3):
        system_prompt, user_prompt = generate_static_score_system_prompt_zh_no_ref, point_wise_score_prompt
        user_prompt = user_prompt.format(task_prompt=question, article=clean_response, criteria_list=rubrics)
        llm_response_str = ""
        for _ in range(max_retries):
            try:
                llm_response_str = await self.llm_client.generate(user_prompt=user_prompt, system_prompt=system_prompt)
                if llm_response_str is not None:
                    break
            except Exception as e:
                logger.error(f"Failed to generate score: {e}")
                return None
            
        if llm_response_str is None:
            logger.error(f"Failed to generate score")
            return None
        # print(f'llm_response_str:{llm_response_str}')
        try:
            json_str_extracted = extract_json_from_markdown(llm_response_str)
            # print(f'json_str_extracted:{json_str_extracted}')
            if json_str_extracted is None:
                raise ValueError("Failed to extract JSON from LLM response")
        except Exception as e:
            logger.error(f"{e}")
            return None
        
        return json_str_extracted
            
    def cal_reward(self, json_str_extracted: dict, static_rubric: dict):
        try:
            json_score = json.loads(json_str_extracted)
            expected = ["comprehensiveness", "insight", "instruction_following", "readability"]
            if not all(d in json_score for d in expected):
                missing = [d for d in expected if d not in json_score]
                logging.error(f"Missing expected dimensions: {missing}")
            
            scores = calculate_weighted_scores(json_score, static_rubric)
            target_total = scores["target"]["total"]
            reference_total = scores["reference"]["total"]
            overall_score = target_total
            return overall_score
        except Exception as e:
            logger.error(f"There are errors in calculating the reward: {e}")
            return 0.0

    async def score_response(self, question: str, response: str, rubrics) -> float:
        try:
            # clean_response = self.clean_text(response)
            clean_response = response

            if clean_response is None:
                return 0.0
            json_str_extracted = await self.generate_score(question, clean_response, rubrics)
            print(json_str_extracted)
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

        # 若缺少 问题、回答、标签 任一种返回零
        if response is None or question_text is None:
            return self.miss_score

        return await self.score_response(question_text, response)

#########################################################################################################3
# 加载 JSONL 文件
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

# 构造 rubric 格式
def build_rubric(entry):
    return {
        "dimension_weight": entry["dimension_weight"],
        "criterions": {
            dim: [
                {
                    "criterion": c["criterion"],
                    "explanation": c["explanation"],
                    "weight": c["weight"]
                }
                for c in crit_list
            ]
            for dim, crit_list in entry["criterions"].items()
        }
    }

# 主执行函数
async def main():
    data_path = "/mnt/workspace/yangguang/exp/4.jsonl"     # 包含 prompt 和 article
    rubric_path = "/mnt/workspace/yangguang/exp/example_criteria.jsonl" # 包含 prompt 和 rubric

    data_list = load_jsonl(data_path)
    rubric_list = load_jsonl(rubric_path)

    # 构建 rubric 映射：prompt → rubric
    rubric_map = {entry["prompt"]: build_rubric(entry) for entry in rubric_list}

    scorer = ReportRewardStrategy()

    for entry in data_list:
        prompt = entry["prompt"]
        article = entry["article"]
        rubric = rubric_map.get(prompt)

        if rubric is None:
            print(f"⚠️ 未找到对应 rubric，跳过 ID {entry['id']}")
            continue

        action = Action(content=article)
        sample = RewardSample(question=prompt)

        score = await scorer.score_response(prompt, article, rubric)
        print(f"✅ ID {entry['id']} 得分: {score:.4f}")

if __name__ == "__main__":
    asyncio.run(main())



# if __name__ == '__main__':
#     fake_action = Action(
#         content=article
#     )
#     fake_sample = RewardSample(
#         question="收集整理目前中国9阶层实际收入和财务状况，特别研究得出中国的中产有哪些特点，实际中产人数，财力等等"
#     )
#     fake_label = article

#     report_reward = ReportRewardStrategy()
#     reward = asyncio.run(report_reward.score(action=fake_action, label=fake_label, sample=fake_sample) )
#     print(reward)
