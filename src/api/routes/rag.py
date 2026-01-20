"""RAG API routes for AgingResearchAI.

Endpoints for knowledge base search and document ingestion.
"""

import logging
import time

from fastapi import APIRouter, HTTPException

from ..dependencies import get_knowledge_base, get_retriever, get_settings
from ..models import (
    RAGSearchRequest,
    RAGSearchResponse,
    RAGDocumentResponse,
    RAGIngestRequest,
    RAGIngestResponse,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/search",
    response_model=RAGSearchResponse,
    responses={
        503: {"model": ErrorResponse, "description": "RAG service unavailable"},
    },
    summary="Search knowledge base",
    description="""
Search the knowledge base using hybrid retrieval (semantic + keyword).

Features:
- PubMedBERT embeddings for biomedical text
- Optional cross-encoder reranking
- Configurable result count and minimum score
    """,
)
async def search_knowledge_base(request: RAGSearchRequest):
    """Search the RAG knowledge base."""
    start_time = time.time()

    retriever = get_retriever(request.collection)
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service unavailable. Knowledge base not initialized.",
        )

    try:
        # Perform search
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            use_reranking=request.use_reranking,
        )

        # Filter by minimum score
        filtered_results = [
            r for r in results
            if r.score >= request.min_score
        ]

        # Format response
        documents = []
        for r in filtered_results:
            doc = r.document
            documents.append(RAGDocumentResponse(
                id=getattr(doc, 'id', ''),
                title=getattr(doc.metadata, 'title', None) if hasattr(doc, 'metadata') else None,
                content=doc.content[:1000] if hasattr(doc, 'content') else str(doc)[:1000],
                source=getattr(doc.metadata, 'source', None) if hasattr(doc, 'metadata') else None,
                pmid=getattr(doc.metadata, 'pmid', None) if hasattr(doc, 'metadata') else None,
                score=r.score,
                metadata=doc.metadata.model_dump() if hasattr(doc.metadata, 'model_dump') else {},
            ))

        elapsed_ms = int((time.time() - start_time) * 1000)

        return RAGSearchResponse(
            query=request.query,
            collection=request.collection,
            total_results=len(documents),
            documents=documents,
            elapsed_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.exception(f"RAG search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}",
        )


@router.post(
    "/ingest",
    response_model=RAGIngestResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "RAG service unavailable"},
    },
    summary="Ingest documents",
    description="""
Ingest documents from various sources into the knowledge base.

Supported sources:
- **pubmed**: Load papers from PubMed using NCBI E-utilities
- **patent**: Load patents from Lens.org
- **pdf**: Load from local PDF files
    """,
)
async def ingest_documents(request: RAGIngestRequest):
    """Ingest documents into the knowledge base."""
    start_time = time.time()

    kb = get_knowledge_base()
    if kb is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service unavailable. Knowledge base not initialized.",
        )

    try:
        documents = []

        if request.source == "pubmed":
            if not request.query:
                raise HTTPException(
                    status_code=400,
                    detail="Query required for PubMed source",
                )

            from src.rag import PubMedLoader

            settings = get_settings()
            loader = PubMedLoader(
                query=request.query,
                max_results=request.max_documents,
                api_key=settings.ncbi_api_key,
            )
            documents = loader.load()

        elif request.source == "patent":
            if not request.query:
                raise HTTPException(
                    status_code=400,
                    detail="Query required for patent source",
                )

            from src.rag import PatentLoader

            loader = PatentLoader(
                query=request.query,
                max_results=request.max_documents,
            )
            documents = loader.load()

        elif request.source == "pdf":
            if not request.pdf_paths:
                raise HTTPException(
                    status_code=400,
                    detail="pdf_paths required for PDF source",
                )

            from src.rag import PDFLoader

            for path in request.pdf_paths[:request.max_documents]:
                try:
                    loader = PDFLoader(path)
                    documents.extend(loader.load())
                except Exception as e:
                    logger.warning(f"Failed to load PDF {path}: {e}")

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported source: {request.source}. Use 'pubmed', 'patent', or 'pdf'.",
            )

        # Add to knowledge base
        if documents:
            kb.add_documents(documents, collection=request.collection)

        elapsed_ms = int((time.time() - start_time) * 1000)

        return RAGIngestResponse(
            source=request.source,
            collection=request.collection,
            documents_added=len(documents),
            elapsed_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Document ingestion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.get(
    "/collections",
    summary="List collections",
    description="List all available collections in the knowledge base.",
)
async def list_collections():
    """List available collections."""
    kb = get_knowledge_base()
    if kb is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service unavailable.",
        )

    try:
        collections = kb.list_collections() if hasattr(kb, 'list_collections') else ["literature"]
        return {
            "collections": collections,
            "default": "literature",
        }
    except Exception as e:
        logger.exception(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/collections/{collection}/stats",
    summary="Get collection statistics",
    description="Get statistics for a specific collection.",
)
async def collection_stats(collection: str):
    """Get collection statistics."""
    kb = get_knowledge_base()
    if kb is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service unavailable.",
        )

    try:
        count = kb.count(collection=collection) if hasattr(kb, 'count') else 0
        return {
            "collection": collection,
            "document_count": count,
        }
    except Exception as e:
        logger.exception(f"Failed to get collection stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/collections/{collection}",
    summary="Delete collection",
    description="Delete a collection from the knowledge base.",
)
async def delete_collection(collection: str):
    """Delete a collection."""
    kb = get_knowledge_base()
    if kb is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service unavailable.",
        )

    if collection == "literature":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete default 'literature' collection.",
        )

    try:
        if hasattr(kb, 'delete_collection'):
            kb.delete_collection(collection)

        return {
            "status": "deleted",
            "collection": collection,
        }
    except Exception as e:
        logger.exception(f"Failed to delete collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
