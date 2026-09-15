import os
from typing import Optional, Dict, Any
import os
from typing import Optional
from openai import OpenAI
from fastmcp import Client
import asyncio
import dotenv

dotenv.load_dotenv()

class AIClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("SEARCH_AGENT_API_KEY")
        self.base_url = base_url or os.getenv("SEARCH_AGENT_BASE_URL")
        self.model = model or os.getenv("SEARCH_AGENT_MODEL")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    async def generate(self, user_prompt: str, system_prompt: str = "", model: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Failed to generate content: {str(e)}")
