import time
from typing import Dict, Any, Optional

class QuotaTracker:
    """Parses real rate-limit headers and tracks available capacity per model."""

    def __init__(self):
        self._quotas: Dict[str, Dict[str, Any]] = {}

    def get_key(self, provider: str, model: str) -> str:
        return f"{provider}:{model}"

    def update_from_headers(self, provider: str, model: str, headers: Dict[str, str]):
        key = self.get_key(provider, model)
        data = self._quotas.setdefault(key, {
            "requests_remaining": 1000,
            "tokens_remaining": 200000,
            "reset_at": time.time() + 60,
            "updated_at": time.time()
        })

        # Standard Groq / Cerebras / OpenAI rate limit headers
        norm_headers = {k.lower(): v for k, v in headers.items()}
        
        # Requests remaining
        for req_hdr in ["x-ratelimit-remaining-requests", "ratelimit-remaining-requests", "x-ratelimit-remaining"]:
            if req_hdr in norm_headers:
                try:
                    data["requests_remaining"] = int(norm_headers[req_hdr])
                    break
                except ValueError:
                    pass

        # Tokens remaining
        for tok_hdr in ["x-ratelimit-remaining-tokens", "ratelimit-remaining-tokens"]:
            if tok_hdr in norm_headers:
                try:
                    data["tokens_remaining"] = int(norm_headers[tok_hdr])
                    break
                except ValueError:
                    pass

        # Reset timestamp
        for reset_hdr in ["x-ratelimit-reset-requests", "ratelimit-reset-requests", "retry-after"]:
            if reset_hdr in norm_headers:
                try:
                    val = float(norm_headers[reset_hdr])
                    data["reset_at"] = time.time() + val if val < 100000 else val
                    break
                except ValueError:
                    pass

        data["updated_at"] = time.time()

    def record_failure(self, provider: str, model: str, is_429: bool = False, retry_after: float = 60.0):
        key = self.get_key(provider, model)
        data = self._quotas.setdefault(key, {
            "requests_remaining": 1000,
            "tokens_remaining": 200000,
            "reset_at": time.time(),
            "updated_at": time.time()
        })
        if is_429:
            data["requests_remaining"] = 0
            data["reset_at"] = time.time() + retry_after

    def get_quota(self, provider: str, model: str) -> Dict[str, Any]:
        key = self.get_key(provider, model)
        return self._quotas.get(key, {
            "requests_remaining": 1000,
            "tokens_remaining": 200000,
            "reset_at": time.time(),
            "updated_at": time.time()
        })
