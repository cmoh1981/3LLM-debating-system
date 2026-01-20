"""FastAPI endpoints for AgingResearchAI.

REST API providing:
- Multi-LLM debate system
- RAG knowledge base search
- ADMET molecular prediction
- Research pipeline execution
"""

from .main import app, run_server
from .dependencies import (
    get_settings,
    get_gemini_client,
    get_deepseek_client,
    get_kimi_client,
    get_claude_client,
    get_openai_client,
    get_knowledge_base,
    get_retriever,
    get_admet_predictor,
    get_debate_engine,
    get_service_status,
)
from .models import (
    HealthResponse,
    DebateRequest,
    DebateResponse,
    RAGSearchRequest,
    RAGSearchResponse,
    ADMETRequest,
    ADMETResponse,
    PipelineRequest,
    PipelineResponse,
    ErrorResponse,
)

__all__ = [
    # Application
    "app",
    "run_server",
    # Dependencies
    "get_settings",
    "get_gemini_client",
    "get_deepseek_client",
    "get_kimi_client",
    "get_claude_client",
    "get_openai_client",
    "get_knowledge_base",
    "get_retriever",
    "get_admet_predictor",
    "get_debate_engine",
    "get_service_status",
    # Models
    "HealthResponse",
    "DebateRequest",
    "DebateResponse",
    "RAGSearchRequest",
    "RAGSearchResponse",
    "ADMETRequest",
    "ADMETResponse",
    "PipelineRequest",
    "PipelineResponse",
    "ErrorResponse",
]
