"""FastAPI application for AgingResearchAI.

REST API providing:
- Multi-LLM debate system
- RAG knowledge base search
- ADMET molecular prediction
- Research pipeline execution
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import get_settings, get_service_status
from .models import (
    HealthResponse,
    ErrorResponse,
)
from .routes import debate, rag, admet, pipeline

logger = logging.getLogger(__name__)


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    # Startup
    logger.info("AgingResearchAI API starting...")
    settings = get_settings()
    logger.info(f"Debug mode: {settings.debug}")

    # Check available services
    status = get_service_status()
    available = [k for k, v in status.items() if v]
    logger.info(f"Available services: {', '.join(available) or 'none'}")

    yield

    # Shutdown
    logger.info("AgingResearchAI API shutting down...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="AgingResearchAI API",
    description="""
Multi-model AI system for aging and metabolic disease drug discovery.

## Features

- **Multi-LLM Debate**: Scientific claims verified by 3 independent LLMs
- **RAG Knowledge Base**: PubMed literature search with semantic retrieval
- **ADMET Prediction**: Molecular property prediction using DeepChem
- **Research Pipeline**: End-to-end pathogenesis and target discovery

## Cost Profile

- Gemini (Proposer): FREE tier
- DeepSeek (Critic): ~$0.14/M tokens
- Kimi (Judge): ~$0.20/M tokens

## Target Diseases

- Type 2 Diabetes (T2D)
- NAFLD/NASH
- Sarcopenia
    """,
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Include Routers
# =============================================================================

app.include_router(debate.router, prefix="/api/v1/debate", tags=["Debate"])
app.include_router(rag.router, prefix="/api/v1/rag", tags=["RAG"])
app.include_router(admet.router, prefix="/api/v1/admet", tags=["ADMET"])
app.include_router(pipeline.router, prefix="/api/v1/pipeline", tags=["Pipeline"])


# =============================================================================
# Root Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """API root - returns health status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        services=get_service_status(),
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    services = get_service_status()

    # Determine overall health
    critical_services = ["gemini"]  # At minimum need proposer
    is_healthy = any(services.get(svc, False) for svc in critical_services)

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        version="1.0.0",
        services=services,
    )


@app.get("/api/v1/status", tags=["Health"])
async def api_status():
    """Detailed API status including model availability."""
    services = get_service_status()

    # Count available debate models
    debate_models = sum(1 for k in ["gemini", "deepseek", "kimi", "claude", "openai"]
                        if services.get(k, False))

    return {
        "status": "operational",
        "services": services,
        "debate_ready": debate_models >= 2,
        "debate_models_available": debate_models,
        "rag_ready": services.get("knowledge_base", False),
        "admet_ready": services.get("admet", False),
    }


@app.get("/api/v1/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get application metrics."""
    try:
        from src.monitoring import get_metrics as get_app_metrics
        metrics = get_app_metrics()
        return metrics.get_summary()
    except ImportError:
        return {"error": "Metrics module not available"}


@app.get("/api/v1/costs", tags=["Monitoring"])
async def get_costs():
    """Get cost tracking information."""
    try:
        from src.monitoring import get_cost_tracker
        tracker = get_cost_tracker()
        return {
            "budget_status": tracker.check_budget(),
            "daily_summary": tracker.get_daily_summary().__dict__,
            "optimization_tips": tracker.get_cost_optimization_tips(),
            "pricing_info": tracker.get_pricing_info(),
        }
    except ImportError:
        return {"error": "Cost tracking module not available"}


@app.get("/api/v1/costs/estimate", tags=["Monitoring"])
async def estimate_cost(model: str, input_tokens: int, output_tokens: int):
    """Estimate cost for a planned API call."""
    try:
        from src.monitoring import quick_cost_estimate
        return quick_cost_estimate(model, input_tokens, output_tokens)
    except ImportError:
        return {"error": "Cost tracking module not available"}


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "detail": str(exc) if get_settings().debug else "An unexpected error occurred",
    }


# =============================================================================
# Run Application
# =============================================================================

def run_server():
    """Run the API server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run_server()
