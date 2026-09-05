import time
import asyncio
import os
import dotenv
import httpx
from typing import Any, Dict
from fastmcp import Client
from openai import AsyncOpenAI
import anyio
import re
from openrlhf_agent.agentkit.tools.base import ToolBase
from deep_research_agent.prompt.post_process import POST_PROCESS_PROMPT
from tool_monitor import ToolMonitor

dotenv.load_dotenv()

monitor = ToolMonitor()

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

    def __init__(self):
        super().__init__()

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        query = arguments.get("query", None)
        if query is None:
            return "query is required"

        topn = max(1, min(int(arguments.get("count", 3)), 10))
        search_remote_url = os.getenv("SEARCH_REMOTE_URL")

        try:
            async with Client(search_remote_url, timeout=20) as client:
                response = await client.call_tool(
                    "baidusearch__search",
                    {"query": query, "count": topn}
                )
                text_list = getattr(response, "content", [])
                context_text = text_list[0].text if text_list else None

                if context_text and context_text.strip():
                    print(f"[BaiduSearch] 成功返回")
                    return str(context_text)
                else:
                    print(f"[BaiduSearch] 返回为空")
                    return ""
        except Exception as e:
            print(f"[BaiduSearch] 异常 {e} ")
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

    _se = asyncio.Semaphore(15)
    _global_vllm_client = AsyncOpenAI(
        base_url=os.getenv('SEARCH_AGENT_BASE_URL'),
        api_key=os.getenv('SEARCH_AGENT_API_KEY'),
    )
    def __init__(self):
        super().__init__()

    async def post_process(self, raw_text: str):
        processed_content = POST_PROCESS_PROMPT.replace("{raw_text}", raw_text)
        try:
            tmp_result = await Crawler360Tool._global_vllm_client.chat.completions.create(
                model=os.getenv('SEARCH_AGENT_MODEL'),
                messages=[{'role': 'user', 'content': processed_content}],
                max_tokens=20000,
                timeout=15,
            )
            return tmp_result.choices[0].message.content
        except Exception as e:
            return None

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        fetch_remote_url = os.getenv("FETCH_REMOTE_URL")
        url = arguments.get("url", None)
        if not url:
            return "url is required"

        tool_key = "crawler"
        await monitor.inc_total(tool_key)
        await monitor.inc_pending(tool_key)
        total_start = time.perf_counter()

        context_text = None

        try:
            async with Crawler360Tool._se:
                await monitor.dec_pending(tool_key)
                await monitor.inc_active(tool_key)

                wait_time = time.perf_counter() - total_start
                fetch_start = time.perf_counter()

                try:
                    async with Client(fetch_remote_url, timeout=20) as client:
                        response = await client.call_tool(
                            "360_crawler",
                            {"url": url}
                        )
                        text_list = getattr(response, "content", [])
                        context_text = text_list[0].text if text_list else None
                        fetch_duration = time.perf_counter() - fetch_start

                    if context_text and len(context_text) > 500:
                        await monitor.inc_crawler_fetch_success()
                        print(f'[Crawler360] 抓取成功 (排队:{wait_time:.2f}s 耗时:{fetch_duration:.2f}s) {await monitor.stats(tool_key)}')
                    else:
                        print(f"[Crawler360] 抓取为空或太短 {await monitor.stats(tool_key)}")
                        return ""

                except Exception as e:
                    print(f"[Crawler360] 抓取异常 {e} {await monitor.stats(tool_key)}")
                    if await monitor._get_active(tool_key) > 0:
                        await monitor.dec_active(tool_key)
                    return ""               
        except Exception as e:
            print(f"[Crawler360] 抓取外层异常 {e} {await monitor.stats(tool_key)}")

        finally:
            if await monitor._get_active(tool_key) > 0:  # ← 安全兜底：仅当 active > 0 才 dec
                await monitor.dec_active(tool_key)
                
        try:
            proc_start = time.perf_counter()
            summary_text = await asyncio.wait_for(
                self.post_process(raw_text=context_text),
                timeout=15.0
            )
            if summary_text and isinstance(summary_text, str) and len(summary_text) > 100:
                await monitor.inc_crawler_post_success()
                print(f"[Crawler360] 后处理成功 (耗时:{time.perf_counter() - proc_start:.2f}s)")
                return summary_text
            else:
                print(f"[Crawler360] 后处理返回无效或过短")
                return str(context_text)[:16000] if context_text else ""
        except Exception as e:
            print(f'[Crawler360] 后处理异常 {e} {await monitor.stats(tool_key)}')
            return str(context_text)[:16000] if context_text else ""

class JinaReaderTool(ToolBase):
    name = "jina_reader"
    description = "Use Jina AI Reader to convert a URL into clean, LLM-friendly Markdown text."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The webpage URL to read and convert"}
        },
        "required": ["url"]
    }
    _semaphore = asyncio.Semaphore(10)

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("JINA_API_KEY", "")

    async def _clean_text(self, text: str) -> str:
        pattern = r"\(https?:.*?\)|\[https?:.*?\]"
        text = re.sub(pattern, "", text)
        text = text.replace('---', '-').replace('===', '=')
        text = re.sub(r' {2,}', ' ', text)
        return text

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        url = arguments.get("url", None)
        if not url:
            return "url is required"

        tool_key = "jina"
        await monitor.inc_total(tool_key)
        await monitor.inc_pending(tool_key)
        total_start = time.perf_counter()
        
        try:
            async with JinaReaderTool._semaphore:
                await monitor.dec_pending(tool_key)
                await monitor.inc_active(tool_key)

                try:
                    headers = {
                        # 'Authorization': f'Bearer {self.api_key}',
                        'X-Return-Format': 'markdown',
                    }
                    fetch_start = time.perf_counter()
                    wait_time = time.perf_counter() - total_start
                    raw_text= ""

                    async with httpx.AsyncClient(timeout=15) as client:
                        response = await client.get(f'https://r.jina.ai/{url}', headers=headers)
                        response.raise_for_status()

                        raw_text = response.text
                        fetch_duration = time.perf_counter() - fetch_start

                    if raw_text and len(raw_text) > 100:
                        content = await self._clean_text(raw_text)
                        await monitor.inc_jina_success()
                        print(f'[JinaReader] 调用成功 (排队:{wait_time:.2f}s 耗时:{fetch_duration:.2f}s) {await monitor.stats(tool_key)}')
                        return content
                    else:
                        print(f"[JinaReader] 返回内容为空或过短 {await monitor.stats(tool_key)}")
                except Exception as e:
                    print(f"[JinaReader] 异常: {repr(e)} {await monitor.stats(tool_key)}")
                    return ""

        except Exception as e:
            print(f"[JinaReader] 外层异常: {str(e)} {await monitor.stats(tool_key)}")
            return ""
            
        finally:
            if await monitor._get_active(tool_key) > 0:
                await monitor.dec_active(tool_key)