from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .schemas import Capability


class WorkerConfig(BaseModel):
    id: str
    capability: Capability
    model: str
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = "local-no-key"
    max_tokens: int = 384
    context_budget: int = 1800
    temperature: float = 0.0
    timeout_seconds: float = 60.0
    system_prompt: str
    enabled: bool = True


class RoutingConfig(BaseModel):
    fast_path: bool = True
    confidence_threshold: float = 0.72
    max_worker_retries: int = 1
    max_manager_replans: int = 1
    max_parallel_steps: int = 3
    verify_all_mutations: bool = True


class ToolPolicyConfig(BaseModel):
    project_root_only: bool = True
    allow_shell: bool = False
    allow_writes: bool = False
    command_timeout_seconds: int = 30
    max_output_chars: int = 12000
    blocked_command_fragments: List[str] = Field(default_factory=lambda: [
        "rm -rf", "remove-item -recurse", "format ", "mkfs", "shutdown", "reboot",
        "stop-computer", "restart-computer", ":(){:|:&};:",
    ])


class ContextConfig(BaseModel):
    path: str = ".dllm/project.cntx"
    max_events_per_package: int = 12
    semantic_compaction: bool = True
    lexical_retrieval: bool = True
    vector_backend: str = "none"
    turboquant_bits: int = 4


class WebRagConfig(BaseModel):
    enabled: bool = False
    searxng_url: Optional[str] = None
    result_limit: int = 3
    fetch_timeout_seconds: float = 10.0
    max_document_chars: int = 12000
    allowed_domains: List[str] = Field(default_factory=list)


class DLLMConfig(BaseModel):
    version: str = "0.1"
    workers: List[WorkerConfig]
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    tools: ToolPolicyConfig = Field(default_factory=ToolPolicyConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    web_rag: WebRagConfig = Field(default_factory=WebRagConfig)

    def workers_by_capability(self) -> Dict[Capability, WorkerConfig]:
        return {worker.capability: worker for worker in self.workers if worker.enabled}


def load_config(path: str | Path) -> DLLMConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return DLLMConfig.model_validate(json.load(handle))

