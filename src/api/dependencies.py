"""FastAPI dependencies for AgingResearchAI.

Provides dependency injection for API routes.
"""

import logging
import os
from functools import lru_cache
from typing import Generator

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class Settings:
    """Application settings from environment."""

    def __init__(self):
        # API Keys
        self.google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
        self.xai_api_key: str | None = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        self.deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
        self.kimi_api_key: str | None = os.getenv("KIMI_API_KEY")
        self.anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
        self.ncbi_api_key: str | None = os.getenv("NCBI_API_KEY")

        # Paths
        self.data_dir: str = os.getenv("DATA_DIR", "data")
        self.embeddings_dir: str = os.path.join(self.data_dir, "embeddings")
        self.models_dir: str = os.getenv("MODELS_DIR", "models")

        # API Settings
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0")
        self.api_port: int = int(os.getenv("API_PORT", "8000"))
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    @property
    def has_gemini(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_grok(self) -> bool:
        return bool(self.xai_api_key)

    @property
    def has_deepseek(self) -> bool:
        return bool(self.deepseek_api_key)

    @property
    def has_kimi(self) -> bool:
        return bool(self.kimi_api_key)

    @property
    def has_claude(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# =============================================================================
# Model Client Dependencies
# =============================================================================

_gemini_client = None
_grok_client = None
_deepseek_client = None
_kimi_client = None
_claude_client = None
_openai_client = None
_qwen_client = None


def get_gemini_client():
    """Get or create Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        settings = get_settings()
        if not settings.has_gemini:
            return None
        try:
            from src.models import GeminiClient
            _gemini_client = GeminiClient(api_key=settings.google_api_key)
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client: {e}")
            return None
    return _gemini_client


def get_grok_client():
    """Get or create Grok client."""
    global _grok_client
    if _grok_client is None:
        settings = get_settings()
        if not settings.has_grok:
            return None
        try:
            from src.models import GrokClient
            _grok_client = GrokClient(api_key=settings.xai_api_key)
        except Exception as e:
            logger.warning(f"Failed to initialize Grok client: {e}")
            return None
    return _grok_client


def get_deepseek_client():
    """Get or create DeepSeek client."""
    global _deepseek_client
    if _deepseek_client is None:
        settings = get_settings()
        if not settings.has_deepseek:
            return None
        try:
            from src.models import DeepSeekClient
            _deepseek_client = DeepSeekClient(api_key=settings.deepseek_api_key)
        except Exception as e:
            logger.warning(f"Failed to initialize DeepSeek client: {e}")
            return None
    return _deepseek_client


def get_kimi_client():
    """Get or create Kimi client."""
    global _kimi_client
    if _kimi_client is None:
        settings = get_settings()
        if not settings.has_kimi:
            return None
        try:
            from src.models import KimiClient
            _kimi_client = KimiClient(api_key=settings.kimi_api_key)
        except Exception as e:
            logger.warning(f"Failed to initialize Kimi client: {e}")
            return None
    return _kimi_client


def get_claude_client():
    """Get or create Claude client."""
    global _claude_client
    if _claude_client is None:
        settings = get_settings()
        if not settings.has_claude:
            return None
        try:
            from src.models import ClaudeClient
            _claude_client = ClaudeClient(api_key=settings.anthropic_api_key)
        except Exception as e:
            logger.warning(f"Failed to initialize Claude client: {e}")
            return None
    return _claude_client


def get_openai_client():
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        settings = get_settings()
        if not settings.has_openai:
            return None
        try:
            from src.models import OpenAIClient
            _openai_client = OpenAIClient(api_key=settings.openai_api_key)
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
            return None
    return _openai_client


def get_available_debate_clients() -> dict:
    """Get all available clients for debate.

    Primary configuration: Gemini (proposer) + Grok (critic) + DeepSeek (judge)
    """
    clients = {}

    # Proposer: Gemini (FREE)
    gemini = get_gemini_client()
    if gemini:
        clients["gemini"] = gemini

    # Critic: Grok (fallback to Claude)
    grok = get_grok_client()
    if grok:
        clients["grok"] = grok
    else:
        claude = get_claude_client()
        if claude:
            clients["claude"] = claude

    # Judge: DeepSeek (fallback to OpenAI)
    deepseek = get_deepseek_client()
    if deepseek:
        clients["deepseek"] = deepseek
    else:
        openai = get_openai_client()
        if openai:
            clients["openai"] = openai

    return clients


# =============================================================================
# RAG Dependencies
# =============================================================================

_knowledge_base = None


def get_knowledge_base():
    """Get or create knowledge base instance."""
    global _knowledge_base
    if _knowledge_base is None:
        settings = get_settings()
        try:
            from src.rag import KnowledgeBase
            _knowledge_base = KnowledgeBase(
                persist_directory=settings.embeddings_dir
            )
        except Exception as e:
            logger.warning(f"Failed to initialize knowledge base: {e}")
            return None
    return _knowledge_base


def get_retriever(collection: str = "literature"):
    """Get retriever for specified collection."""
    kb = get_knowledge_base()
    if kb is None:
        return None
    try:
        from src.rag import HybridRetriever
        return HybridRetriever(kb, collection=collection)
    except Exception as e:
        logger.warning(f"Failed to create retriever: {e}")
        return None


# =============================================================================
# ADMET Dependencies
# =============================================================================

_admet_predictor = None


def get_admet_predictor():
    """Get or create ADMET predictor."""
    global _admet_predictor
    if _admet_predictor is None:
        try:
            from src.admet import DeepChemPredictor
            settings = get_settings()
            _admet_predictor = DeepChemPredictor(
                model_dir=settings.models_dir
            )
        except ImportError as e:
            logger.warning(f"DeepChem not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize ADMET predictor: {e}")
            return None
    return _admet_predictor


# =============================================================================
# Debate Engine Dependencies
# =============================================================================

def get_debate_engine(config: dict | None = None):
    """Create debate engine with available clients."""
    clients = get_available_debate_clients()
    if len(clients) < 2:
        logger.warning("Insufficient clients for debate (need at least 2)")
        return None

    try:
        from src.chains import DebateEngine, DebateConfig

        debate_config = DebateConfig(**(config or {}))
        return DebateEngine(clients=clients, config=debate_config)
    except Exception as e:
        logger.warning(f"Failed to create debate engine: {e}")
        return None


# =============================================================================
# Service Status
# =============================================================================

def get_service_status() -> dict[str, bool]:
    """Check status of all services."""
    return {
        "gemini": get_gemini_client() is not None,
        "grok": get_grok_client() is not None,
        "deepseek": get_deepseek_client() is not None,
        "kimi": get_kimi_client() is not None,
        "claude": get_claude_client() is not None,
        "openai": get_openai_client() is not None,
        "knowledge_base": get_knowledge_base() is not None,
        "admet": get_admet_predictor() is not None,
    }
