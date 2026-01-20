"""API request/response models for AgingResearchAI.

Pydantic models for API validation and serialization.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Health Check
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    services: dict[str, bool] = Field(default_factory=dict)


# =============================================================================
# Debate API Models
# =============================================================================

class ClaimInput(BaseModel):
    """Input claim for debate."""

    text: str = Field(..., min_length=10, description="Claim text")
    proposer: str = Field(default="user", description="Who proposed this claim")
    evidence: list[str] = Field(default_factory=list, description="Initial evidence")


class DebateRequest(BaseModel):
    """Request to start a debate session."""

    topic: str = Field(..., min_length=5, description="Topic to debate")
    claims: list[ClaimInput] = Field(default_factory=list, description="Initial claims")
    context: str = Field(default="", description="Background context (e.g., from RAG)")
    max_rounds: int = Field(default=3, ge=1, le=5, description="Maximum debate rounds")
    consensus_threshold: float = Field(default=0.66, ge=0.5, le=1.0)


class DebateClaimResponse(BaseModel):
    """Single claim result from debate."""

    text: str
    proposer: str
    confidence: float
    votes: dict[str, str]
    critiques_count: int


class DebateResponse(BaseModel):
    """Response from debate session."""

    topic: str
    rounds: int
    consensus_claims: list[DebateClaimResponse]
    rejected_claims: list[DebateClaimResponse]
    unresolved_claims: list[DebateClaimResponse]
    overall_confidence: float
    debate_summary: str
    elapsed_time_ms: int = 0


# =============================================================================
# RAG API Models
# =============================================================================

class RAGSearchRequest(BaseModel):
    """Request for RAG search."""

    query: str = Field(..., min_length=3, description="Search query")
    collection: str = Field(default="literature", description="Collection to search")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    use_reranking: bool = Field(default=True, description="Apply cross-encoder reranking")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum relevance score")


class RAGDocumentResponse(BaseModel):
    """Single document from RAG search."""

    id: str
    title: str | None
    content: str
    source: str | None
    pmid: str | None
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGSearchResponse(BaseModel):
    """Response from RAG search."""

    query: str
    collection: str
    total_results: int
    documents: list[RAGDocumentResponse]
    elapsed_time_ms: int = 0


class RAGIngestRequest(BaseModel):
    """Request to ingest documents into RAG."""

    source: str = Field(..., description="pubmed, patent, or pdf")
    query: str = Field(default="", description="Query for PubMed/patent search")
    max_documents: int = Field(default=50, ge=1, le=500)
    collection: str = Field(default="literature")
    pdf_paths: list[str] = Field(default_factory=list, description="PDF file paths for pdf source")


class RAGIngestResponse(BaseModel):
    """Response from document ingestion."""

    source: str
    collection: str
    documents_added: int
    elapsed_time_ms: int = 0


# =============================================================================
# ADMET API Models
# =============================================================================

class ADMETRequest(BaseModel):
    """Request for ADMET prediction."""

    smiles: str = Field(..., description="SMILES string of compound")
    compound_id: str | None = Field(default=None, description="Optional compound identifier")
    quick_check: bool = Field(default=False, description="Use quick RDKit-only check")


class ADMETBatchRequest(BaseModel):
    """Batch ADMET prediction request."""

    compounds: list[dict[str, str]] = Field(
        ...,
        description="List of {smiles, compound_id} dicts"
    )
    quick_check: bool = Field(default=False)


class PhysicochemicalResponse(BaseModel):
    """Physicochemical properties."""

    molecular_weight: float
    logp: float
    hbd: int
    hba: int
    tpsa: float
    rotatable_bonds: int


class LipinskiResponse(BaseModel):
    """Lipinski rule assessment."""

    passes: bool
    violations: int
    details: dict[str, bool]


class ADMETResponse(BaseModel):
    """Full ADMET prediction response."""

    smiles: str
    compound_id: str | None
    physicochemical: PhysicochemicalResponse
    lipinski: LipinskiResponse
    overall_risk: str
    flags: list[str]
    recommendations: list[str]
    absorption: dict[str, Any]
    distribution: dict[str, Any]
    metabolism: dict[str, Any]
    excretion: dict[str, Any]
    toxicity: dict[str, Any]
    elapsed_time_ms: int = 0


class QuickADMETResponse(BaseModel):
    """Quick ADMET check response."""

    smiles: str
    molecular_weight: float
    logp: float
    hbd: int
    hba: int
    tpsa: float
    rotatable_bonds: int
    lipinski_violations: int
    lipinski_passes: bool
    estimated_absorption: str
    estimated_bbb: str
    overall_risk: str


# =============================================================================
# Image Analysis Models (Qwen-VL)
# =============================================================================

class ImageAnalysisRequest(BaseModel):
    """Request for image analysis."""

    image_path: str = Field(..., description="Path to image file")
    analysis_type: str = Field(
        default="general",
        description="Type: microscopy, western_blot, or general"
    )
    additional_context: str = Field(default="", description="Additional analysis context")


class ImageAnalysisResponse(BaseModel):
    """Response from image analysis."""

    analysis_type: str
    summary: str
    findings: list[str]
    confidence: float
    elapsed_time_ms: int = 0


# =============================================================================
# Pipeline API Models
# =============================================================================

class PipelineRequest(BaseModel):
    """Request to run full research pipeline."""

    disease: str = Field(..., description="Disease to research (T2D, NAFLD, Sarcopenia)")
    tissue: str = Field(default="liver", description="Target tissue")
    workflow_type: str = Field(
        default="pathogenesis",
        description="Workflow: pathogenesis, target_discovery, or full"
    )
    enable_debate: bool = Field(default=True, description="Enable multi-LLM debate")
    use_rag: bool = Field(default=True, description="Use RAG for context")


class PipelineResponse(BaseModel):
    """Response from research pipeline."""

    run_id: str
    disease: str
    tissue: str
    workflow_type: str
    status: str
    summary: str
    claims: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    debate_results: dict[str, Any] | None
    next_actions: list[dict[str, Any]]
    elapsed_time_ms: int = 0


# =============================================================================
# Error Models
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
