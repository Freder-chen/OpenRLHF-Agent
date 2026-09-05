from typing import Any, Dict
from fastmcp import Client
import dotenv
import os
import asyncio
from agents import AsyncOpenAI
from openrlhf_agent.agentkit.tools.base import ToolBase
from deep_research_agent.prompt.post_process import POST_PROCESS_PROMPT

dotenv.load_dotenv()

class BaiduSearchTool(ToolBase):
    name = "baidu_search"
    description = "Use Baidu Search to obtain web page results"
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keyword"},
            "count": {"type": "integer", "description": "Number of results to return"}
        },
        "required": ["query", "count"]
    }

    def __init__(self, max_concurrency: int = 3):
        super().__init__()
        self.se = asyncio.Semaphore(max_concurrency)

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        query = arguments.get("query", None)
        if query is None:
            return "query is required"

        topn = max(1, min(int(arguments.get("topn", 3)), 10))

        search_remote_url = os.getenv("SEARCH_REMOTE_URL")

        try:
            async with self.se:
                async with Client(search_remote_url, timeout=30) as client:
                    response = await client.call_tool(
                        "baidusearch__search",
                        {"query": query, "count": topn}
                    )
                    text_list = getattr(response, "content", [])
                    context_text = text_list[0].text if text_list else None
                    if not context_text.strip():
                        print(f"[BaiduSearch] 返回结果为空")
                        return ""
                    else:
                        print('[BaiduSearch] 成功返回')
                        return str(context_text)

        except asyncio.TimeoutError:
            print(f"[BaiduSearch] 超时")
            return ""
        except ConnectionError as e:
            print(f"[BaiduSearch] 连接失败")
            return ""
        except Exception as e:
            print(f"[BaiduSearch] 其他异常{e}")
            return ""


class Crawler360Tool(ToolBase):
    name = "crawler_360"
    description = "Use 360 crawler to fetch web page results"
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The webpage URL to fetch"}
        },
        "required": ["url"]
    }

    global_vllm_client = AsyncOpenAI(
        base_url=os.getenv('SEARCH_AGENT_BASE_URL'),
        api_key=os.getenv('SEARCH_AGENT_API_KEY'),
    )

    def __init__(self, max_concurrency: int = 2):
        super().__init__()
        self.se = asyncio.Semaphore(max_concurrency)
        self.vllm_client = AsyncOpenAI(
            base_url=os.getenv('SEARCH_AGENT_BASE_URL'),
            api_key=os.getenv('SEARCH_AGENT_API_KEY'),
        )    

    async def post_process(self, raw_text: str):
        tmp_result = await Crawler360Tool.global_vllm_client.chat.completions.create(
            model=os.getenv('SEARCH_AGENT_MODEL'),
            messages=[{'role': 'user', 'content': POST_PROCESS_PROMPT.replace("{raw_text}", raw_text)}],
            max_tokens=20000
        )
        return tmp_result.choices[0].message.content

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        fetch_remote_url = os.getenv("FETCH_REMOTE_URL")
        url = arguments.get("url", None)
        if not url:
            return "url is required"

        try:
            async with self.se:
                async with Client(fetch_remote_url, timeout=45) as client:
                    response = await client.call_tool(
                        "360_crawler",
                        {"url": url}
                    )
                    
                    text_list = getattr(response, "content", [])
                    context_text = text_list[0].text if text_list else None
                    if context_text and len(context_text) > 500:
                        print('[Crawler360] 成功返回')
                        try:
                            summary_text = await asyncio.wait_for(
                                self.post_process(raw_text=context_text),
                                timeout=45.0
                            )
                            if summary_text and len(summary_text) > 100:
                                print(f"[Crawler360] 后处理成功")
                                return summary_text
                            else:
                                print(f"[Crawler360] 后处理返回结果为空")
                                return str(context_text[: 16000])
                        except asyncio.TimeoutError:
                            print(f"[Crawler360] 后处理超时")
                            return str(context_text[: 16000])
                        except Exception as e:
                            print(f'[Crawler360] LLM后处理异常{e}')
                            return str(context_text[:16000])

                    else:
                        print(f"[Crawler360] 返回结果为空")
                        return "" 

        except asyncio.TimeoutError:
            print(f"[Crawler360] 超时")
            return ""
        except ConnectionError:
            print(f"[Crawler360] 连接失败")
            return ""
        except Exception as e:
            print(f"[Crawler360] 其他异常{e}")
            return ""
