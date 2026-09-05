from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Protocol, Type

import httpx
from pydantic import BaseModel

from .config import WorkerConfig


class ModelTransport(Protocol):
    model_calls: int
    manager_calls: int

    async def generate(self, worker: WorkerConfig, payload: Dict[str, Any], response_model: Type[BaseModel]) -> BaseModel:
        ...


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


class LlamaCppTransport:
    """OpenAI-compatible transport for independent llama.cpp worker endpoints.

    The same base URL may point to llama-server router mode. In that mode the
    `model` field selects which GGUF is loaded, allowing phase-gated residency.
    """

    def __init__(self):
        self.model_calls = 0
        self.manager_calls = 0
        self.total_latency_ms = 0.0
        self._clients: Dict[tuple[str, str, float], httpx.AsyncClient] = {}

    def _client(self, worker: WorkerConfig) -> httpx.AsyncClient:
        key = (worker.base_url, worker.api_key, worker.timeout_seconds)
        client = self._clients.get(key)
        if client is None:
            client = httpx.AsyncClient(
                timeout=worker.timeout_seconds,
                headers={"Authorization": f"Bearer {worker.api_key}"},
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            )
            self._clients[key] = client
        return client

    async def generate(self, worker: WorkerConfig, payload: Dict[str, Any], response_model: Type[BaseModel]) -> BaseModel:
        schema = response_model.model_json_schema()
        body = {
            "model": worker.model,
            "messages": [
                {"role": "system", "content": worker.system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            ],
            "temperature": worker.temperature,
            "max_tokens": worker.max_tokens,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": response_model.__name__, "strict": True, "schema": schema},
            },
        }
        started = time.perf_counter()
        response = await self._client(worker).post(
            f"{worker.base_url.rstrip('/')}/chat/completions",
            json=body,
        )
        response.raise_for_status()
        self.total_latency_ms += (time.perf_counter() - started) * 1000
        self.model_calls += 1
        if worker.capability.value == "manager":
            self.manager_calls += 1
        content = response.json()["choices"][0]["message"]["content"]
        return response_model.model_validate(_extract_json(content))

    async def health(self, worker: WorkerConfig) -> bool:
        try:
            response = await self._client(worker).get(
                f"{worker.base_url.rstrip('/')}/models",
                timeout=min(worker.timeout_seconds, 5.0),
            )
            return response.is_success
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
