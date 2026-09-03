import time
import logging
from typing import List, Dict, Optional, Tuple
from .models import ModelState, TaskRequirements, TaskType, ChatRequest, ChatResponse
from .quota import QuotaTracker
from .adapters.base import BaseProvider

logger = logging.getLogger("broker.router")

class ModelRouter:
    """Intelligent scheduler that scores models dynamically instead of round-robin."""

    def __init__(self, quota_tracker: QuotaTracker):
        self.quota_tracker = quota_tracker
        self.providers: Dict[str, BaseProvider] = {}
        self.models: List[ModelState] = []
        
        # Adaptive learning weights
        self.weights = {
            "quality": 0.35,
            "quota": 0.20,
            "reliability": 0.20,
            "capability": 0.15,
            "latency": 0.10
        }

    def register_provider(self, provider: BaseProvider):
        self.providers[provider.name] = provider

    def register_model(self, model: ModelState):
        self.models.append(model)

    def calculate_route_score(self, model: ModelState, req: TaskRequirements) -> float:
        # Check quota status from quota tracker
        quota = self.quota_tracker.get_quota(model.provider, model.model)
        remaining_reqs = quota.get("requests_remaining", model.requests_remaining)
        reset_at = quota.get("reset_at", model.reset_at)

        # 1. Hard constraints
        if not model.healthy or (remaining_reqs <= 1 and time.time() < reset_at):
            return -1.0
        
        if req.requires_tools and not model.capabilities.tools:
            return -1.0
            
        if req.requires_json and not model.capabilities.json_mode:
            return -1.0
            
        if req.estimated_tokens > model.context_length:
            return -1.0

        # 2. Dynamic scoring
        quota_ratio = min(1.0, max(0.0, remaining_reqs / float(model.max_daily_requests or 1000)))
        speed_score = 1.0 / (1.0 + max(0.1, model.latency_ema))

        # Capability matching
        cap_score = 0.5
        if req.task == TaskType.CODING and model.capabilities.coding:
            cap_score = 1.0
        elif req.task == TaskType.REASONING and model.capabilities.reasoning:
            cap_score = 0.95
        elif req.task == TaskType.CRITIC and model.capabilities.reasoning:
            cap_score = 0.90
        elif req.task == TaskType.SUMMARIZATION:
            cap_score = 0.85

        score = (
            model.quality_score * self.weights["quality"] +
            quota_ratio * self.weights["quota"] +
            model.success_ema * self.weights["reliability"] +
            cap_score * self.weights["capability"] +
            speed_score * self.weights["latency"]
        )

        # Penalty for recent consecutive failures
        score -= (model.consecutive_failures * 0.25)
        return max(0.0, score)

    def select_best_model(self, req: TaskRequirements) -> List[Tuple[ModelState, float]]:
        ranked = []
        for m in self.models:
            score = self.calculate_route_score(m, req)
            if score > 0:
                ranked.append((m, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    async def route_and_execute(self, request: ChatRequest) -> ChatResponse:
        reqs = request.requirements or TaskRequirements(task=request.task)
        ranked = self.select_best_model(reqs)

        if not ranked:
            raise RuntimeError("No available healthy models matching request requirements.")

        errors = []
        for model_state, score in ranked:
            provider = self.providers.get(model_state.provider)
            if not provider:
                continue

            try:
                logger.info(f"Routing task '{reqs.task}' to {model_state.provider}/{model_state.model} (score: {score:.2f})")
                t0 = time.time()
                response = await provider.chat(model_state.model, request)
                duration = time.time() - t0

                # Update EMA metrics on success
                model_state.latency_ema = 0.8 * model_state.latency_ema + 0.2 * duration
                model_state.success_ema = min(1.0, 0.9 * model_state.success_ema + 0.1)
                model_state.consecutive_failures = 0
                return response

            except Exception as e:
                logger.warning(f"Provider {model_state.provider}/{model_state.model} failed: {e}. Trying fallback...")
                model_state.consecutive_failures += 1
                model_state.success_ema = max(0.1, 0.8 * model_state.success_ema)
                errors.append(f"{model_state.provider}: {str(e)}")
                continue

        raise RuntimeError(f"All routed candidates failed: {'; '.join(errors)}")
