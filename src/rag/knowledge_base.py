"""Knowledge Base management for AgingResearchAI.

Provides ChromaDB-based vector storage for scientific literature,
patents, and other research documents.
"""

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

# ChromaDB import with fallback
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None


# =============================================================================
# Data Models
# =============================================================================

class DocumentMetadata(BaseModel):
    """Metadata for a document in the knowledge base."""

    source: str  # "pubmed", "patent", "pdf", "database"
    source_id: str | None = None  # PMID, patent number, etc.
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    publication_date: str | None = None
    journal: str | None = None
    doi: str | None = None
    url: str | None = None

    # For aging research
    disease_tags: list[str] = Field(default_factory=list)
    gene_mentions: list[str] = Field(default_factory=list)
    pathway_mentions: list[str] = Field(default_factory=list)
    drug_mentions: list[str] = Field(default_factory=list)

    # Quality indicators
    citation_count: int | None = None
    evidence_tier: str | None = None  # tier1, tier2, tier3

    # Indexing info
    indexed_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    chunk_index: int = 0
    total_chunks: int = 1


class Document(BaseModel):
    """A document or document chunk in the knowledge base."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    metadata: DocumentMetadata
    embedding: list[float] | None = None


class SearchResult(BaseModel):
    """Result from a knowledge base search."""

    document: Document
    score: float  # Similarity score (0-1, higher is better)
    rank: int


class CollectionStats(BaseModel):
    """Statistics for a collection."""

    name: str
    document_count: int
    sources: dict[str, int]
    diseases: dict[str, int]
    last_updated: str | None = None


# =============================================================================
# Knowledge Base
# =============================================================================

class KnowledgeBase:
    """ChromaDB-based knowledge base for scientific literature.

    Supports:
    - Multiple collections (literature, patents, databases)
    - Semantic search with embeddings
    - Metadata filtering
    - Batch operations

    Usage:
        kb = KnowledgeBase(persist_directory="data/embeddings")
        kb.add_documents(documents, collection="literature")
        results = kb.search("AMPK signaling in diabetes", top_k=10)
    """

    DEFAULT_COLLECTIONS = [
        "literature",   # PubMed papers
        "patents",      # Patent documents
        "databases",    # Database entries (DrugBank, KEGG, etc.)
        "experiments",  # Experimental results
    ]

    def __init__(
        self,
        persist_directory: str | Path = "data/embeddings",
        embedding_function: Any = None,
    ):
        """Initialize knowledge base.

        Args:
            persist_directory: Directory to persist ChromaDB data
            embedding_function: Custom embedding function (uses default if None)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB not installed. Run: pip install chromadb"
            )

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )

        self.embedding_function = embedding_function
        self._collections: dict[str, Any] = {}

    def get_or_create_collection(self, name: str) -> Any:
        """Get or create a collection.

        Args:
            name: Collection name

        Returns:
            ChromaDB collection
        """
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
        return self._collections[name]

    def add_documents(
        self,
        documents: list[Document],
        collection: str = "literature",
    ) -> list[str]:
        """Add documents to the knowledge base.

        Args:
            documents: List of documents to add
            collection: Collection name

        Returns:
            List of document IDs
        """
        coll = self.get_or_create_collection(collection)

        ids = []
        contents = []
        metadatas = []
        embeddings = []

        for doc in documents:
            ids.append(doc.id)
            contents.append(doc.content)
            metadatas.append(doc.metadata.model_dump())
            if doc.embedding:
                embeddings.append(doc.embedding)

        # Add to collection
        if embeddings and len(embeddings) == len(documents):
            coll.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        else:
            coll.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
            )

        return ids

    def search(
        self,
        query: str,
        collection: str = "literature",
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> list[SearchResult]:
        """Search the knowledge base.

        Args:
            query: Search query
            collection: Collection to search
            top_k: Number of results to return
            filter_metadata: Metadata filters
            include_embeddings: Whether to include embeddings in results

        Returns:
            List of SearchResult objects
        """
        coll = self.get_or_create_collection(collection)

        # Build query parameters
        query_params = {
            "query_texts": [query],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filter_metadata:
            query_params["where"] = filter_metadata

        if include_embeddings:
            query_params["include"].append("embeddings")

        # Execute search
        results = coll.query(**query_params)

        # Convert to SearchResult objects
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, (doc_id, content, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                # Convert distance to similarity score (cosine distance -> similarity)
                score = 1 - distance

                doc = Document(
                    id=doc_id,
                    content=content,
                    metadata=DocumentMetadata(**metadata),
                    embedding=results.get("embeddings", [[]])[0][i] if include_embeddings else None,
                )

                search_results.append(SearchResult(
                    document=doc,
                    score=score,
                    rank=i + 1,
                ))

        return search_results

    def search_by_metadata(
        self,
        collection: str = "literature",
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[Document]:
        """Search by metadata without semantic query.

        Args:
            collection: Collection to search
            filters: Metadata filters
            limit: Maximum results

        Returns:
            List of matching documents
        """
        coll = self.get_or_create_collection(collection)

        results = coll.get(
            where=filters,
            limit=limit,
            include=["documents", "metadatas"],
        )

        documents = []
        if results and results["ids"]:
            for doc_id, content, metadata in zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
            ):
                documents.append(Document(
                    id=doc_id,
                    content=content,
                    metadata=DocumentMetadata(**metadata),
                ))

        return documents

    def get_document(
        self,
        doc_id: str,
        collection: str = "literature",
    ) -> Document | None:
        """Get a document by ID.

        Args:
            doc_id: Document ID
            collection: Collection name

        Returns:
            Document or None if not found
        """
        coll = self.get_or_create_collection(collection)

        results = coll.get(
            ids=[doc_id],
            include=["documents", "metadatas"],
        )

        if results and results["ids"]:
            return Document(
                id=results["ids"][0],
                content=results["documents"][0],
                metadata=DocumentMetadata(**results["metadatas"][0]),
            )

        return None

    def delete_documents(
        self,
        doc_ids: list[str],
        collection: str = "literature",
    ) -> int:
        """Delete documents by ID.

        Args:
            doc_ids: List of document IDs
            collection: Collection name

        Returns:
            Number of documents deleted
        """
        coll = self.get_or_create_collection(collection)
        coll.delete(ids=doc_ids)
        return len(doc_ids)

    def update_document(
        self,
        doc_id: str,
        content: str | None = None,
        metadata: DocumentMetadata | None = None,
        collection: str = "literature",
    ) -> bool:
        """Update a document.

        Args:
            doc_id: Document ID
            content: New content (optional)
            metadata: New metadata (optional)
            collection: Collection name

        Returns:
            True if updated successfully
        """
        coll = self.get_or_create_collection(collection)

        update_params = {"ids": [doc_id]}
        if content:
            update_params["documents"] = [content]
        if metadata:
            update_params["metadatas"] = [metadata.model_dump()]

        coll.update(**update_params)
        return True

    def get_collection_stats(self, collection: str = "literature") -> CollectionStats:
        """Get statistics for a collection.

        Args:
            collection: Collection name

        Returns:
            CollectionStats object
        """
        coll = self.get_or_create_collection(collection)
        count = coll.count()

        # Get source distribution
        sources: dict[str, int] = {}
        diseases: dict[str, int] = {}

        if count > 0:
            # Sample documents to get distribution
            sample = coll.get(limit=min(count, 1000), include=["metadatas"])
            if sample and sample["metadatas"]:
                for meta in sample["metadatas"]:
                    source = meta.get("source", "unknown")
                    sources[source] = sources.get(source, 0) + 1

                    for disease in meta.get("disease_tags", []):
                        diseases[disease] = diseases.get(disease, 0) + 1

        return CollectionStats(
            name=collection,
            document_count=count,
            sources=sources,
            diseases=diseases,
            last_updated=datetime.now().isoformat(),
        )

    def list_collections(self) -> list[str]:
        """List all collections.

        Returns:
            List of collection names
        """
        collections = self.client.list_collections()
        return [c.name for c in collections]

    def clear_collection(self, collection: str) -> bool:
        """Clear all documents from a collection.

        Args:
            collection: Collection name

        Returns:
            True if cleared
        """
        try:
            self.client.delete_collection(collection)
            if collection in self._collections:
                del self._collections[collection]
            return True
        except Exception:
            return False

    def search_similar(
        self,
        doc_id: str,
        collection: str = "literature",
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Find documents similar to a given document.

        Args:
            doc_id: ID of reference document
            collection: Collection name
            top_k: Number of similar documents

        Returns:
            List of similar documents
        """
        # Get the reference document's embedding
        coll = self.get_or_create_collection(collection)

        ref_doc = coll.get(
            ids=[doc_id],
            include=["embeddings"],
        )

        if not ref_doc or not ref_doc.get("embeddings"):
            return []

        # Search by embedding
        results = coll.query(
            query_embeddings=ref_doc["embeddings"],
            n_results=top_k + 1,  # +1 to exclude self
            include=["documents", "metadatas", "distances"],
        )

        # Convert to SearchResult, excluding the reference document
        search_results = []
        for i, (result_id, content, metadata, distance) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            if result_id == doc_id:
                continue

            score = 1 - distance
            doc = Document(
                id=result_id,
                content=content,
                metadata=DocumentMetadata(**metadata),
            )
            search_results.append(SearchResult(
                document=doc,
                score=score,
                rank=len(search_results) + 1,
            ))

            if len(search_results) >= top_k:
                break

        return search_results
