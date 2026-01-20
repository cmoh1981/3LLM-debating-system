"""Document retrieval for AgingResearchAI RAG system.

Provides multiple retrieval strategies:
- Semantic retrieval (embedding similarity)
- Keyword retrieval (BM25-style)
- Hybrid retrieval (combines both)
- Reranking for improved relevance
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from .knowledge_base import Document, KnowledgeBase, SearchResult
from .embeddings import EmbeddingModel, compute_similarity


# =============================================================================
# Configuration
# =============================================================================

class RetrievalConfig(BaseModel):
    """Configuration for retrieval."""

    top_k: int = 10
    similarity_threshold: float = 0.5
    rerank: bool = True
    rerank_top_k: int = 5

    # Hybrid weights
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3

    # Filters
    max_age_days: int | None = None
    required_sources: list[str] = Field(default_factory=list)
    disease_filter: list[str] = Field(default_factory=list)


# =============================================================================
# Base Retriever
# =============================================================================

class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant documents.

        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters

        Returns:
            List of SearchResult objects
        """
        pass

    @abstractmethod
    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with scores.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of (Document, score) tuples
        """
        pass


# =============================================================================
# Semantic Retriever
# =============================================================================

class SemanticRetriever(BaseRetriever):
    """Retriever using semantic similarity (embeddings).

    Uses ChromaDB vector search for fast similarity matching.

    Usage:
        retriever = SemanticRetriever(knowledge_base)
        results = retriever.retrieve("AMPK in diabetes", top_k=10)
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        collection: str = "literature",
        config: RetrievalConfig | None = None,
    ):
        """Initialize semantic retriever.

        Args:
            knowledge_base: Knowledge base to search
            collection: Collection name
            config: Retrieval configuration
        """
        self.kb = knowledge_base
        self.collection = collection
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve documents using semantic similarity."""
        k = top_k or self.config.top_k

        results = self.kb.search(
            query=query,
            collection=self.collection,
            top_k=k,
            filter_metadata=filters,
        )

        # Filter by similarity threshold
        filtered = [
            r for r in results
            if r.score >= self.config.similarity_threshold
        ]

        return filtered

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with scores."""
        results = self.retrieve(query, top_k)
        return [(r.document, r.score) for r in results]


# =============================================================================
# Keyword Retriever
# =============================================================================

class KeywordRetriever(BaseRetriever):
    """Retriever using keyword matching (simple TF-IDF style).

    For cases where exact keyword matches are important.

    Usage:
        retriever = KeywordRetriever(knowledge_base)
        results = retriever.retrieve("SIRT1 NAD+", top_k=10)
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        collection: str = "literature",
    ):
        """Initialize keyword retriever.

        Args:
            knowledge_base: Knowledge base to search
            collection: Collection name
        """
        self.kb = knowledge_base
        self.collection = collection

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve documents using keyword matching.

        Note: This is a simplified implementation.
        For production, consider using Elasticsearch or similar.
        """
        # Split query into keywords
        keywords = query.lower().split()

        # Get all documents (with limit)
        # In production, this would use a proper inverted index
        all_docs = self.kb.search_by_metadata(
            collection=self.collection,
            filters=filters,
            limit=1000,
        )

        # Score documents by keyword matches
        scored_docs = []
        for doc in all_docs:
            content_lower = doc.content.lower()
            score = sum(
                content_lower.count(kw) / len(content_lower)
                for kw in keywords
            )
            if score > 0:
                scored_docs.append((doc, score))

        # Sort by score and take top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = scored_docs[:top_k]

        # Convert to SearchResult
        results = [
            SearchResult(
                document=doc,
                score=score,
                rank=i + 1,
            )
            for i, (doc, score) in enumerate(top_docs)
        ]

        return results

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with scores."""
        results = self.retrieve(query, top_k)
        return [(r.document, r.score) for r in results]


# =============================================================================
# Hybrid Retriever
# =============================================================================

class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining semantic and keyword search.

    Uses weighted combination of both approaches for better results.

    Usage:
        retriever = HybridRetriever(knowledge_base)
        results = retriever.retrieve("AMPK signaling pathway", top_k=10)
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        collection: str = "literature",
        config: RetrievalConfig | None = None,
    ):
        """Initialize hybrid retriever.

        Args:
            knowledge_base: Knowledge base to search
            collection: Collection name
            config: Retrieval configuration
        """
        self.kb = knowledge_base
        self.collection = collection
        self.config = config or RetrievalConfig()

        self.semantic_retriever = SemanticRetriever(
            knowledge_base, collection, config
        )
        self.keyword_retriever = KeywordRetriever(
            knowledge_base, collection
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve using hybrid approach."""
        k = top_k or self.config.top_k

        # Get more results from each retriever
        fetch_k = k * 2

        # Semantic search
        semantic_results = self.semantic_retriever.retrieve(
            query, fetch_k, filters
        )

        # Keyword search
        keyword_results = self.keyword_retriever.retrieve(
            query, fetch_k, filters
        )

        # Combine scores using Reciprocal Rank Fusion (RRF)
        doc_scores: dict[str, tuple[Document, float]] = {}

        # Add semantic results
        for result in semantic_results:
            doc_id = result.document.id
            rrf_score = self.config.semantic_weight / (result.rank + 60)
            doc_scores[doc_id] = (result.document, rrf_score)

        # Add keyword results
        for result in keyword_results:
            doc_id = result.document.id
            rrf_score = self.config.keyword_weight / (result.rank + 60)
            if doc_id in doc_scores:
                # Combine scores
                existing = doc_scores[doc_id]
                doc_scores[doc_id] = (existing[0], existing[1] + rrf_score)
            else:
                doc_scores[doc_id] = (result.document, rrf_score)

        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        # Convert to SearchResult
        results = [
            SearchResult(
                document=doc,
                score=score,
                rank=i + 1,
            )
            for i, (doc, score) in enumerate(sorted_docs)
        ]

        return results

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with scores."""
        results = self.retrieve(query, top_k)
        return [(r.document, r.score) for r in results]


# =============================================================================
# Reranker
# =============================================================================

class CrossEncoderReranker:
    """Reranker using cross-encoder for improved relevance.

    Cross-encoders are more accurate than bi-encoders but slower,
    so we use them to rerank a smaller set of candidates.

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, results, top_k=5)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """Initialize reranker.

        Args:
            model_name: Cross-encoder model name
        """
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.available = True
        except ImportError:
            self.model = None
            self.available = False

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank results using cross-encoder.

        Args:
            query: Search query
            results: Initial results to rerank
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if not self.available or not results:
            return results[:top_k]

        # Create query-document pairs
        pairs = [(query, r.document.content) for r in results]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Combine with original scores
        reranked = []
        for result, ce_score in zip(results, scores):
            # Weighted combination of original and cross-encoder score
            combined_score = 0.3 * result.score + 0.7 * float(ce_score)
            reranked.append((result.document, combined_score))

        # Sort by combined score
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Convert back to SearchResult
        return [
            SearchResult(
                document=doc,
                score=score,
                rank=i + 1,
            )
            for i, (doc, score) in enumerate(reranked[:top_k])
        ]


# =============================================================================
# Context Builder
# =============================================================================

class ContextBuilder:
    """Build context from retrieved documents for LLM prompts.

    Formats retrieved documents into a context string suitable
    for inclusion in LLM prompts.

    Usage:
        builder = ContextBuilder()
        context = builder.build_context(results, max_tokens=4000)
    """

    def __init__(
        self,
        include_metadata: bool = True,
        include_citations: bool = True,
        max_doc_length: int = 500,
    ):
        """Initialize context builder.

        Args:
            include_metadata: Include source info
            include_citations: Include PMIDs/DOIs
            max_doc_length: Max characters per document
        """
        self.include_metadata = include_metadata
        self.include_citations = include_citations
        self.max_doc_length = max_doc_length

    def build_context(
        self,
        results: list[SearchResult],
        max_tokens: int = 4000,
    ) -> str:
        """Build context string from search results.

        Args:
            results: Search results
            max_tokens: Approximate max tokens (4 chars/token)

        Returns:
            Formatted context string
        """
        max_chars = max_tokens * 4
        context_parts = []
        total_chars = 0

        for result in results:
            doc = result.document
            meta = doc.metadata

            # Format document
            parts = []

            if self.include_metadata and meta.title:
                parts.append(f"**{meta.title}**")

            if self.include_citations:
                citation_parts = []
                if meta.source_id:
                    citation_parts.append(f"{meta.source.upper()}: {meta.source_id}")
                if meta.doi:
                    citation_parts.append(f"DOI: {meta.doi}")
                if citation_parts:
                    parts.append(f"[{', '.join(citation_parts)}]")

            # Truncate content if needed
            content = doc.content
            if len(content) > self.max_doc_length:
                content = content[:self.max_doc_length] + "..."

            parts.append(content)

            # Add relevance score
            parts.append(f"(Relevance: {result.score:.2f})")

            doc_text = "\n".join(parts)

            # Check if we can fit this document
            if total_chars + len(doc_text) > max_chars:
                break

            context_parts.append(doc_text)
            total_chars += len(doc_text) + 2  # +2 for separator

        return "\n\n---\n\n".join(context_parts)

    def build_citation_list(
        self,
        results: list[SearchResult],
    ) -> list[dict[str, str]]:
        """Build a list of citations from results.

        Args:
            results: Search results

        Returns:
            List of citation dictionaries
        """
        citations = []
        for result in results:
            meta = result.document.metadata
            citation = {
                "source": meta.source,
                "id": meta.source_id or "",
                "title": meta.title or "",
                "doi": meta.doi or "",
            }
            if meta.authors:
                citation["authors"] = ", ".join(meta.authors[:3])
                if len(meta.authors) > 3:
                    citation["authors"] += " et al."
            citations.append(citation)
        return citations
