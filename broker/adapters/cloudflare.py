import time
import httpx
from typing import Dict, Any
from .base import BaseProvider
from ..models import ChatRequest, ChatResponse
from ..quota import QuotaTracker

class CloudflareProvider(BaseProvider):
    def __init__(self, api_key: str, account_id: str, quota_tracker: QuotaTracker):
        super().__init__(api_key, "cloudflare")
        self.account_id = account_id
        self.quota_tracker = quota_tracker

    async def chat(self, model: str, request: ChatRequest) -> ChatResponse:
        t0 = time.time()
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": request.max_tokens
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            latency = time.time() - t0

            if resp.status_code == 429:
                self.quota_tracker.record_failure("cloudflare", model, is_429=True)
                raise RuntimeError(f"Cloudflare 429 Rate Limit Exceeded on {model}")

            resp.raise_for_status()
            data = resp.json()

            content = data.get("result", {}).get("response", "")
            return ChatResponse(
                id=f"cf-{int(time.time())}",
                model=model,
                provider="cloudflare",
                content=content,
                usage={"neurons": 1},
                latency_seconds=latency
            )

    async def health(self) -> bool:
        return bool(self.api_key and self.account_id)
