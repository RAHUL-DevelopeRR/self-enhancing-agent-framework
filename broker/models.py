from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import time

class TaskType(str, Enum):
    CODING = "coding"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    CRITIC = "critic"
    GENERAL = "general"
    EXTRACTION = "extraction"

class ModelCapabilities(BaseModel):
    reasoning: bool = True
    coding: bool = True
    vision: bool = False
    tools: bool = True
    json_mode: bool = True

class ModelState(BaseModel):
    provider: str
    model: str
    requests_remaining: int = 1000
    max_daily_requests: int = 1000
    tokens_remaining: int = 200000
    reset_at: float = Field(default_factory=time.time)
    latency_ema: float = 0.5  # in seconds
    success_ema: float = 0.98
    quality_score: float = 0.85
    context_length: int = 32768
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    healthy: bool = True
    consecutive_failures: int = 0

class TaskRequirements(BaseModel):
    task: TaskType = TaskType.GENERAL
    requires_tools: bool = False
    requires_json: bool = False
    requires_vision: bool = False
    estimated_tokens: int = 2000
    min_quality: float = 0.6

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    task: TaskType = TaskType.GENERAL
    role: Optional[str] = None
    requirements: Optional[TaskRequirements] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False

class ChatResponse(BaseModel):
    id: str
    model: str
    provider: str
    content: str
    usage: Dict[str, int] = Field(default_factory=dict)
    latency_seconds: float
    cached: bool = False
