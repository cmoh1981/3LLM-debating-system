"""Integration tests for RAG (Retrieval-Augmented Generation) system.

Tests verify that RAG components work together:
- KnowledgeBase (ChromaDB storage)
- Embeddings (PubMedBERT)
- Retrievers (semantic, keyword, hybrid)
- Context building for LLM prompts
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


# Skip all tests if ChromaDB not available
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
class TestKnowledgeBase:
    """Test KnowledgeBase functionality."""

    def test_knowledge_base_init(self, tmp_path):
        """Test knowledge base initialization."""
        from src.rag.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(persist_directory=tmp_path / "test_kb")
        assert kb is not None
        assert kb.client is not None

    def test_add_and_search_documents(self, tmp_path):
        """Test adding and searching documents."""
        from src.rag.knowledge_base import KnowledgeBase, Document, DocumentMetadata

        kb = KnowledgeBase(persist_directory=tmp_path / "test_kb")

        # Create test documents
        docs = [
            Document(
                id="doc1",
                content="AMPK signaling plays a crucial role in diabetes pathogenesis.",
                metadata=DocumentMetadata(
                    source="pubmed",
                    source_id="12345678",
                    title="AMPK in Diabetes",
                    disease_tags=["Type 2 Diabetes"],
                ),
            ),
            Document(
                id="doc2",
                content="Insulin resistance is mediated by multiple pathways.",
                metadata=DocumentMetadata(
                    source="pubmed",
                    source_id="23456789",
                    title="Insulin Resistance Mechanisms",
                    disease_tags=["Type 2 Diabetes"],
                ),
            ),
        ]

        # Add documents
        ids = kb.add_documents(docs, collection="test_literature")
        assert len(ids) == 2

        # Search
        results = kb.search(
            query="AMPK diabetes signaling",
            collection="test_literature",
            top_k=5,
        )

        assert len(results) > 0
        assert results[0].score > 0

    def test_collection_stats(self, tmp_path):
        """Test collection statistics."""
        from src.rag.knowledge_base import KnowledgeBase, Document, DocumentMetadata

        kb = KnowledgeBase(persist_directory=tmp_path / "test_kb")

        # Add test document
        doc = Document(
            id="test_doc",
            content="Test content for statistics verification.",
            metadata=DocumentMetadata(source="pubmed", source_id="111"),
        )
        kb.add_documents([doc], collection="stats_test")

        # Get stats
        stats = kb.get_collection_stats("stats_test")
        assert stats.document_count == 1
        assert "pubmed" in stats.sources

    def test_delete_documents(self, tmp_path):
        """Test document deletion."""
        from src.rag.knowledge_base import KnowledgeBase, Document, DocumentMetadata

        kb = KnowledgeBase(persist_directory=tmp_path / "test_kb")

        doc = Document(
            id="delete_me",
            content="This document will be deleted.",
            metadata=DocumentMetadata(source="test"),
        )
        kb.add_documents([doc], collection="delete_test")

        # Verify added
        assert kb.get_document("delete_me", "delete_test") is not None

        # Delete
        deleted = kb.delete_documents(["delete_me"], collection="delete_test")
        assert deleted == 1


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
class TestRetrievers:
    """Test retriever implementations."""

    def test_semantic_retriever(self, tmp_path):
        """Test semantic retriever."""
        from src.rag.knowledge_base import KnowledgeBase, Document, DocumentMetadata
        from src.rag.retriever import SemanticRetriever

        kb = KnowledgeBase(persist_directory=tmp_path / "test_kb")

        # Add test documents
        docs = [
            Document(
                id="sem_doc1",
                content="Metformin activates AMPK to reduce hepatic glucose production.",
                metadata=DocumentMetadata(source="pubmed"),
            ),
        ]
        kb.add_documents(docs, collection="semantic_test")

        # Create retriever
        retriever = SemanticRetriever(kb, collection="semantic_test")

        # Retrieve
        results = retriever.retrieve("AMPK activation in liver", top_k=5)
        assert len(results) > 0

    def test_hybrid_retriever(self, tmp_path):
        """Test hybrid retriever combines semantic and keyword search."""
        from src.rag.knowledge_base import KnowledgeBase, Document, DocumentMetadata
        from src.rag.retriever import HybridRetriever

        kb = KnowledgeBase(persist_directory=tmp_path / "test_kb")

        # Add test documents
        docs = [
            Document(
                id="hyb_doc1",
                content="SIRT1 is a key regulator of cellular metabolism and aging.",
                metadata=DocumentMetadata(source="pubmed"),
            ),
            Document(
                id="hyb_doc2",
                content="NAD+ levels decline with age, affecting SIRT1 activity.",
                metadata=DocumentMetadata(source="pubmed"),
            ),
        ]
        kb.add_documents(docs, collection="hybrid_test")

        # Create hybrid retriever
        retriever = HybridRetriever(kb, collection="hybrid_test")

        # Retrieve
        results = retriever.retrieve("SIRT1 NAD+ aging", top_k=5)
        assert len(results) > 0


class TestContextBuilder:
    """Test context building for LLM prompts."""

    def test_build_context(self):
        """Test context string building."""
        from src.rag.knowledge_base import Document, DocumentMetadata, SearchResult
        from src.rag.retriever import ContextBuilder

        # Create mock results
        results = [
            SearchResult(
                document=Document(
                    id="ctx1",
                    content="AMPK activation improves insulin sensitivity.",
                    metadata=DocumentMetadata(
                        source="pubmed",
                        source_id="12345",
                        title="AMPK and Insulin",
                    ),
                ),
                score=0.9,
                rank=1,
            ),
        ]

        builder = ContextBuilder()
        context = builder.build_context(results, max_tokens=1000)

        assert "AMPK" in context
        assert "12345" in context  # PMID should be included

    def test_build_citation_list(self):
        """Test citation list building."""
        from src.rag.knowledge_base import Document, DocumentMetadata, SearchResult
        from src.rag.retriever import ContextBuilder

        results = [
            SearchResult(
                document=Document(
                    id="cit1",
                    content="Test content",
                    metadata=DocumentMetadata(
                        source="pubmed",
                        source_id="11111",
                        title="Test Paper",
                        authors=["Smith J", "Doe J"],
                        doi="10.1000/test",
                    ),
                ),
                score=0.8,
                rank=1,
            ),
        ]

        builder = ContextBuilder()
        citations = builder.build_citation_list(results)

        assert len(citations) == 1
        assert citations[0]["source"] == "pubmed"
        assert citations[0]["id"] == "11111"
        assert "Smith" in citations[0]["authors"]


class TestReranker:
    """Test cross-encoder reranker."""

    def test_reranker_without_model(self):
        """Test reranker handles missing model gracefully."""
        from src.rag.retriever import CrossEncoderReranker

        # Should not crash if sentence-transformers not installed
        reranker = CrossEncoderReranker()

        if not reranker.available:
            # Test fallback behavior
            from src.rag.knowledge_base import Document, DocumentMetadata, SearchResult

            results = [
                SearchResult(
                    document=Document(
                        id="r1",
                        content="Test",
                        metadata=DocumentMetadata(source="test"),
                    ),
                    score=0.9,
                    rank=1,
                ),
            ]

            reranked = reranker.rerank("test query", results, top_k=5)
            assert len(reranked) == 1  # Should return original results


class TestRetrievalConfig:
    """Test retrieval configuration."""

    def test_default_config(self):
        """Test default retrieval configuration."""
        from src.rag.retriever import RetrievalConfig

        config = RetrievalConfig()
        assert config.top_k == 10
        assert config.similarity_threshold == 0.5
        assert config.semantic_weight == 0.7

    def test_custom_config(self):
        """Test custom retrieval configuration."""
        from src.rag.retriever import RetrievalConfig

        config = RetrievalConfig(
            top_k=20,
            similarity_threshold=0.7,
            disease_filter=["T2D", "NAFLD"],
        )
        assert config.top_k == 20
        assert len(config.disease_filter) == 2
