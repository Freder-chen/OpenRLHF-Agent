import asyncio
import dotenv

dotenv.load_dotenv()

class ToolMonitor:
    def __init__(self):
        self._lock = asyncio.Lock()
        self.pending_count = {"crawler": 0, "jina": 0}
        self.active_count = {"crawler": 0, "jina": 0}

        self.crawler_total = 0
        self.crawler_fetch_success = 0
        self.crawler_post_success = 0

        self.jina_total = 0
        self.jina_success = 0

    async def inc_total(self, tool_name: str):
        async with self._lock:
            if tool_name == "crawler":
                self.crawler_total += 1
            elif tool_name == "jina":
                self.jina_total += 1

    async def inc_pending(self, tool_name: str):
        async with self._lock:
            self.pending_count[tool_name] += 1

    async def dec_pending(self, tool_name: str):
        async with self._lock:
            if self.pending_count[tool_name] > 0:
                self.pending_count[tool_name] -= 1

    async def inc_active(self, tool_name: str):
        async with self._lock:
            self.active_count[tool_name] += 1

    async def dec_active(self, tool_name: str):
        async with self._lock:
            if self.active_count[tool_name] > 0:
                self.active_count[tool_name] -= 1

    async def inc_crawler_fetch_success(self):
        async with self._lock:
            self.crawler_fetch_success += 1

    async def inc_crawler_post_success(self):
        async with self._lock:
            self.crawler_post_success += 1

    async def inc_jina_success(self):
        async with self._lock:
            self.jina_success += 1

    async def stats(self, tool_name: str) -> str:
        async with self._lock:
            if tool_name == "jina":
                ratio = f"成功率:{(self.jina_success/self.jina_total*100):.1f}%" if self.jina_total > 0 else "成功率:0%"
                return f"| 等待:{self.pending_count[tool_name]} 处理中:{self.active_count[tool_name]} | {ratio} | {self.jina_total} {self.jina_success}"
            elif tool_name == 'crawler':
                fetch_ratio = f"抓取:{(self.crawler_fetch_success/self.crawler_total*100):.1f}%" if self.crawler_total > 0 else "抓取:0%"
                post_ratio = f"后处理:{(self.crawler_post_success/self.crawler_total*100):.1f}%" if self.crawler_total > 0 else "后处理:0%"
                return f"| 等待:{self.pending_count[tool_name]} 处理中:{self.active_count[tool_name]} | {fetch_ratio} {post_ratio} | {self.crawler_total} {self.crawler_fetch_success} {self.crawler_post_success}"
    
    async def _get_active(self, tool_name: str) -> int:
        async with self._lock:
            return self.active_count[tool_name]
        