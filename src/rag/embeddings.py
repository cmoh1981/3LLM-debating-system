"""Embedding models for AgingResearchAI RAG system.

Provides embedding generation for scientific text using:
- Sentence Transformers (default)
- PubMedBERT (specialized for biomedical text)
- Custom embedding models
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

# Sentence Transformers import with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


# =============================================================================
# Base Embedding Model
# =============================================================================

class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""

    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    max_seq_length: int = 512
    batch_size: int = 32
    normalize: bool = True
    device: str = "cpu"  # "cpu", "cuda", "mps"


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a search query (may use different encoding).

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


# =============================================================================
# Sentence Transformer Embeddings
# =============================================================================

class SentenceTransformerEmbeddings(EmbeddingModel):
    """Embedding model using Sentence Transformers.

    Default model: all-MiniLM-L6-v2 (fast, good quality)
    Alternative: all-mpnet-base-v2 (higher quality, slower)

    Usage:
        embeddings = SentenceTransformerEmbeddings()
        vector = embeddings.embed_text("AMPK signaling in diabetes")
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        """Initialize Sentence Transformer embeddings.

        Args:
            config: Embedding configuration
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

        self.config = config or EmbeddingConfig()
        self.model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
        )
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts with batching."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query."""
        # For symmetric models, query embedding is the same
        return self.embed_text(query)

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


# =============================================================================
# PubMedBERT Embeddings
# =============================================================================

class PubMedBERTEmbeddings(EmbeddingModel):
    """Embedding model using PubMedBERT for biomedical text.

    Optimized for scientific/biomedical literature.
    Model: pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb

    Usage:
        embeddings = PubMedBERTEmbeddings()
        vector = embeddings.embed_text("Type 2 diabetes pathogenesis")
    """

    # PubMedBERT model for semantic similarity
    DEFAULT_MODEL = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
    ):
        """Initialize PubMedBERT embeddings.

        Args:
            model_name: Model name (uses default if None)
            device: Device to use
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device

        self.model = SentenceTransformer(self.model_name, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query."""
        return self.embed_text(query)

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


# =============================================================================
# ChromaDB Embedding Function Wrapper
# =============================================================================

class ChromaEmbeddingFunction:
    """Wrapper to use our embedding models with ChromaDB.

    Usage:
        embedding_fn = ChromaEmbeddingFunction(PubMedBERTEmbeddings())
        collection = client.create_collection(
            name="literature",
            embedding_function=embedding_fn
        )
    """

    def __init__(self, embedding_model: EmbeddingModel):
        """Initialize wrapper.

        Args:
            embedding_model: Embedding model to use
        """
        self.model = embedding_model

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate embeddings for ChromaDB.

        Args:
            input: List of texts

        Returns:
            List of embeddings
        """
        return self.model.embed_texts(input)


# =============================================================================
# Utility Functions
# =============================================================================

def get_embedding_model(
    model_type: str = "default",
    device: str = "cpu",
) -> EmbeddingModel:
    """Get an embedding model by type.

    Args:
        model_type: "default", "pubmedbert", or model name
        device: Device to use

    Returns:
        Initialized embedding model
    """
    if model_type == "default" or model_type == "minilm":
        return SentenceTransformerEmbeddings(
            EmbeddingConfig(
                model_name="all-MiniLM-L6-v2",
                device=device,
            )
        )
    elif model_type == "pubmedbert":
        return PubMedBERTEmbeddings(device=device)
    elif model_type == "mpnet":
        return SentenceTransformerEmbeddings(
            EmbeddingConfig(
                model_name="all-mpnet-base-v2",
                device=device,
            )
        )
    else:
        # Assume it's a model name
        return SentenceTransformerEmbeddings(
            EmbeddingConfig(
                model_name=model_type,
                device=device,
            )
        )


def compute_similarity(
    embedding1: list[float],
    embedding2: list[float],
) -> float:
    """Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding
        embedding2: Second embedding

    Returns:
        Cosine similarity (0-1)
    """
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    # Cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def batch_compute_similarities(
    query_embedding: list[float],
    document_embeddings: list[list[float]],
) -> list[float]:
    """Compute similarities between a query and multiple documents.

    Args:
        query_embedding: Query embedding
        document_embeddings: List of document embeddings

    Returns:
        List of similarity scores
    """
    query_vec = np.array(query_embedding)
    doc_matrix = np.array(document_embeddings)

    # Normalize
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    doc_norms = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-9)

    # Compute similarities
    similarities = np.dot(doc_norms, query_norm)

    return similarities.tolist()
