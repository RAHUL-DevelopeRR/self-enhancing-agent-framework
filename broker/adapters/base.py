from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ..models import ChatRequest, ChatResponse

class BaseProvider(ABC):
    """Abstract base class for all LLM provider adapters."""

    def __init__(self, api_key: str, name: str):
        self.api_key = api_key
        self.name = name

    @abstractmethod
    async def chat(self, model: str, request: ChatRequest) -> ChatResponse:
        """Sends chat request to provider and returns standardized ChatResponse."""
        pass

    @abstractmethod
    async def health(self) -> bool:
        """Checks if provider is reachable and active."""
        pass
