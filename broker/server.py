import os
import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from .models import ChatRequest, ChatResponse, ModelState, ModelCapabilities, TaskType
from .quota import QuotaTracker
from .router import ModelRouter
from .adapters.groq import GroqProvider
from .adapters.cerebras import CerebrasProvider
from .adapters.openrouter import OpenRouterProvider
from .adapters.cloudflare import CloudflareProvider
from .adapters.gemini import GeminiProvider

quota_tracker = QuotaTracker()
router = ModelRouter(quota_tracker)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize providers if API keys are set in environment
    if groq_key := os.getenv("GROQ_API_KEY"):
        router.register_provider(GroqProvider(groq_key, quota_tracker))
        router.register_model(ModelState(
            provider="groq",
            model="llama-3.3-70b-versatile",
            max_daily_requests=1000,
            quality_score=0.88,
            context_length=128000,
            capabilities=ModelCapabilities(coding=True, reasoning=True, json_mode=True, tools=True)
        ))

    if cerebras_key := os.getenv("CEREBRAS_API_KEY"):
        router.register_provider(CerebrasProvider(cerebras_key, quota_tracker))
        router.register_model(ModelState(
            provider="cerebras",
            model="llama3.1-70b",
            max_daily_requests=14400,
            quality_score=0.91,
            context_length=8192,
            capabilities=ModelCapabilities(coding=True, reasoning=True, json_mode=True, tools=True)
        ))

    if openrouter_key := os.getenv("OPENROUTER_API_KEY"):
        router.register_provider(OpenRouterProvider(openrouter_key, quota_tracker))
        router.register_model(ModelState(
            provider="openrouter",
            model="meta-llama/llama-3.3-70b-instruct:free",
            max_daily_requests=50,
            quality_score=0.86,
            context_length=65536,
            capabilities=ModelCapabilities(coding=True, reasoning=True, json_mode=True, tools=True)
        ))

    if gemini_key := os.getenv("GEMINI_API_KEY"):
        router.register_provider(GeminiProvider(gemini_key, quota_tracker))
        router.register_model(ModelState(
            provider="gemini",
            model="gemini-1.5-flash",
            max_daily_requests=1500,
            quality_score=0.92,
            context_length=1000000,
            capabilities=ModelCapabilities(coding=True, reasoning=True, json_mode=True, tools=True, vision=True)
        ))

    if cf_key := os.getenv("CLOUDFLARE_API_TOKEN"):
        cf_account = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        if cf_account:
            router.register_provider(CloudflareProvider(cf_key, cf_account, quota_tracker))
            router.register_model(ModelState(
                provider="cloudflare",
                model="@cf/meta/llama-3.1-8b-instruct",
                max_daily_requests=5000,
                quality_score=0.78,
                context_length=16384,
                capabilities=ModelCapabilities(coding=False, reasoning=True, json_mode=True, tools=False)
            ))
    yield

app = FastAPI(title="Free-Provider AI Broker", version="1.0.0", lifespan=lifespan)

@app.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = await router.route_and_execute(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "registered_models": len(router.models),
        "providers": list(router.providers.keys())
    }

if __name__ == "__main__":
    uvicorn.run("broker.server:app", host="0.0.0.0", port=8000, reload=True)
