"""AgingResearchAI - Multi-model AI system for aging and metabolic disease drug discovery.

A cost-effective multi-LLM system for drug discovery in aging and metabolic diseases,
featuring:
- 3-LLM Debate System (Gemini + DeepSeek + Kimi)
- RAG Knowledge Base (ChromaDB + PubMedBERT)
- ADMET Molecular Prediction (DeepChem + RDKit)
- LangGraph Agent Orchestration

Target Diseases:
- Type 2 Diabetes (T2D)
- NAFLD/NASH
- Sarcopenia

Cost Profile: ~$1-5/day using free/cheap LLM providers.
"""

__version__ = "0.1.0"
__author__ = "AgingResearchAI Team"

# Lazy imports to avoid loading all dependencies on import
def __getattr__(name: str):
    """Lazy import of submodules."""
    if name == "core":
        from . import core
        return core
    elif name == "models":
        from . import models
        return models
    elif name == "chains":
        from . import chains
        return chains
    elif name == "agents":
        from . import agents
        return agents
    elif name == "rag":
        from . import rag
        return rag
    elif name == "admet":
        from . import admet
        return admet
    elif name == "genomics":
        from . import genomics
        return genomics
    elif name == "api":
        from . import api
        return api
    elif name == "monitoring":
        from . import monitoring
        return monitoring
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "__author__",
    # Submodules (lazy loaded)
    "core",
    "models",
    "chains",
    "agents",
    "rag",
    "admet",
    "genomics",
    "api",
    "monitoring",
]
