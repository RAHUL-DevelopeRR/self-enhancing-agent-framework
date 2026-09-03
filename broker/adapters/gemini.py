import time
import httpx
from typing import Dict, Any
from .base import BaseProvider
from ..models import ChatRequest, ChatResponse
from ..quota import QuotaTracker

class GeminiProvider(BaseProvider):
    def __init__(self, api_key: str, quota_tracker: QuotaTracker):
        super().__init__(api_key, "gemini")
        self.quota_tracker = quota_tracker

    async def chat(self, model: str, request: ChatRequest) -> ChatResponse:
        t0 = time.time()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        # Convert messages to Gemini format
        contents = []
        for m in request.messages:
            role = "user" if m.role in ["user", "system"] else "model"
            contents.append({"role": role, "parts": [{"text": m.content}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens
            }
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            latency = time.time() - t0

            if resp.status_code == 429:
                self.quota_tracker.record_failure("gemini", model, is_429=True)
                raise RuntimeError(f"Gemini 429 Rate Limit Exceeded on {model}")

            resp.raise_for_status()
            data = resp.json()

            candidates = data.get("candidates", [])
            content = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                content = "".join([p.get("text", "") for p in parts])

            return ChatResponse(
                id=f"gemini-{int(time.time())}",
                model=model,
                provider="gemini",
                content=content,
                usage=data.get("usageMetadata", {}),
                latency_seconds=latency
            )

    async def health(self) -> bool:
        return bool(self.api_key)
