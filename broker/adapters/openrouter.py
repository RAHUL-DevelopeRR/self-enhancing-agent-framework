import time
import httpx
from typing import Dict, Any
from .base import BaseProvider
from ..models import ChatRequest, ChatResponse
from ..quota import QuotaTracker

class OpenRouterProvider(BaseProvider):
    def __init__(self, api_key: str, quota_tracker: QuotaTracker):
        super().__init__(api_key, "openrouter")
        self.quota_tracker = quota_tracker
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    async def chat(self, model: str, request: ChatRequest) -> ChatResponse:
        t0 = time.time()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/agent-framework",
            "X-Title": "SelfEnhancingAgent"
        }
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(self.base_url, headers=headers, json=payload)
            latency = time.time() - t0

            self.quota_tracker.update_from_headers("openrouter", model, dict(resp.headers))

            if resp.status_code == 429:
                self.quota_tracker.record_failure("openrouter", model, is_429=True)
                raise RuntimeError(f"OpenRouter 429 Rate Limit Exceeded on {model}")

            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            content = choice["message"]["content"]
            usage = data.get("usage", {})

            return ChatResponse(
                id=data.get("id", f"or-{int(time.time())}"),
                model=model,
                provider="openrouter",
                content=content,
                usage=usage,
                latency_seconds=latency
            )

    async def health(self) -> bool:
        return bool(self.api_key)
