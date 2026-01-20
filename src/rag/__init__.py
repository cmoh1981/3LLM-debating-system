"""RAG (Retrieval-Augmented Generation) module for AgingResearchAI.

Provides knowledge base management, document embedding, and retrieval
for augmenting LLM responses with relevant scientific literature.
"""

from .knowledge_base import (
    KnowledgeBase,
    Document,
    DocumentMetadata,
    SearchResult,
)
from .embeddings import (
    EmbeddingModel,
    PubMedBERTEmbeddings,
    get_embedding_model,
)
from .retriever import (
    HybridRetriever,
    SemanticRetriever,
    KeywordRetriever,
    RetrievalConfig,
)
from .document_loaders import (
    PubMedLoader,
    PatentLoader,
    PDFLoader,
    load_documents_from_directory,
)

__all__ = [
    # Knowledge Base
    "KnowledgeBase",
    "Document",
    "DocumentMetadata",
    "SearchResult",
    # Embeddings
    "EmbeddingModel",
    "PubMedBERTEmbeddings",
    "get_embedding_model",
    # Retrievers
    "HybridRetriever",
    "SemanticRetriever",
    "KeywordRetriever",
    "RetrievalConfig",
    # Loaders
    "PubMedLoader",
    "PatentLoader",
    "PDFLoader",
    "load_documents_from_directory",
]
