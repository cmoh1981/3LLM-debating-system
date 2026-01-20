"""Global JSON schema definitions and validators for AgingResearchAI.

All module outputs must conform to this schema for consistency and traceability.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class EvidenceTier(str, Enum):
    """Evidence quality tiers based on LongevityBench findings.

    Tier 1: Replicated + Causal support + Literature evidence
    Tier 2: Replicated association
    Tier 3: Single analysis only (needs validation)
    """

    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"


class EvidenceType(str, Enum):
    """Types of evidence that support claims."""

    LITERATURE = "literature"
    DATABASE = "database"
    COMPUTED = "computed"


class Evidence(BaseModel):
    """Evidence supporting a claim.

    Every claim MUST have at least one evidence entry.
    No uncited claims are allowed in the system.
    """

    type: EvidenceType
    pmid: str | None = Field(None, description="PubMed ID for literature evidence")
    quote: str | None = Field(None, description="Relevant quote from the source")
    source: str | None = Field(None, description="Database name for database evidence")
    id: str | None = Field(None, description="Database entry ID")
    tool: str | None = Field(None, description="Tool name for computed evidence")
    artifact_id: str | None = Field(None, description="Linked artifact ID for computed results")

    @field_validator("pmid")
    @classmethod
    def validate_pmid(cls, v: str | None, info) -> str | None:
        if info.data.get("type") == EvidenceType.LITERATURE and not v:
            raise ValueError("PMID required for literature evidence")
        return v

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str | None, info) -> str | None:
        if info.data.get("type") == EvidenceType.DATABASE and not v:
            raise ValueError("Source required for database evidence")
        return v


class Claim(BaseModel):
    """A scientific claim with evidence and confidence.

    Following LongevityBench principles:
    - Classification over regression (confidence is categorical assessment)
    - Every claim must have evidence
    - Evidence tier indicates quality
    """

    text: str = Field(..., min_length=10, description="The claim statement")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    evidence_tier: EvidenceTier
    evidence: list[Evidence] = Field(..., min_length=1, description="Supporting evidence (required)")

    @field_validator("evidence")
    @classmethod
    def validate_evidence_not_empty(cls, v: list[Evidence]) -> list[Evidence]:
        if not v:
            raise ValueError("Every claim must have at least one evidence entry (no uncited claims)")
        return v


class Provenance(BaseModel):
    """Provenance tracking for reproducibility."""

    code_version: str | None = Field(None, description="Git SHA or version")
    inputs: list[str] = Field(default_factory=list, description="Input artifact IDs")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters used")


class ArtifactType(str, Enum):
    """Types of artifacts produced by modules."""

    TABLE = "table"
    FIGURE = "figure"
    CODE = "code"
    FILE = "file"


class Artifact(BaseModel):
    """An artifact produced by a module (table, figure, code, file)."""

    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    type: ArtifactType
    path: str = Field(..., description="Path to the artifact file")
    provenance: Provenance = Field(default_factory=Provenance)


class Priority(str, Enum):
    """Priority levels for next actions."""

    P0 = "P0"  # Critical
    P1 = "P1"  # Important
    P2 = "P2"  # Nice to have


class NextAction(BaseModel):
    """A suggested next action after module execution."""

    action: str
    priority: Priority
    reason: str


class ErrorInfo(BaseModel):
    """Error information for failed operations."""

    code: str
    message: str


class ModuleStatus(str, Enum):
    """Status of module execution."""

    OK = "ok"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class ModelType(str, Enum):
    """LLM models available in the system."""

    GEMINI = "gemini"
    CLAUDE = "claude"
    LOBSTER = "lobster"


class ModuleOutput(BaseModel):
    """Global output schema for all modules.

    Every module in the system must return this schema.
    This ensures consistency, traceability, and evidence-based claims.
    """

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    module: str = Field(..., description="Module name that produced this output")
    model_used: ModelType
    status: ModuleStatus
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    summary: str = Field(..., min_length=10, description="Human-readable summary")

    claims: list[Claim] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    next_actions: list[NextAction] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
    errors: list[ErrorInfo] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "abc123",
                "module": "pathogenesis",
                "model_used": "claude",
                "status": "ok",
                "timestamp": "2026-01-20T10:00:00",
                "summary": "Identified 3 key pathways in T2D pathogenesis",
                "claims": [
                    {
                        "text": "AMPK signaling is downregulated in T2D liver samples",
                        "confidence": 0.85,
                        "evidence_tier": "tier1",
                        "evidence": [
                            {
                                "type": "literature",
                                "pmid": "12345678",
                                "quote": "AMPK activity was reduced by 40%..."
                            }
                        ]
                    }
                ]
            }
        }


def validate_output(data: dict) -> ModuleOutput:
    """Validate output data against the global schema.

    Args:
        data: Dictionary containing module output

    Returns:
        Validated ModuleOutput object

    Raises:
        ValidationError: If data doesn't conform to schema
    """
    return ModuleOutput.model_validate(data)


def create_output(
    module: str,
    model_used: ModelType,
    summary: str,
    claims: list[dict] | None = None,
    artifacts: list[dict] | None = None,
    status: ModuleStatus = ModuleStatus.OK,
    warnings: list[str] | None = None,
    errors: list[dict] | None = None,
    next_actions: list[dict] | None = None,
) -> ModuleOutput:
    """Helper to create a properly structured module output.

    Args:
        module: Name of the module
        model_used: Which LLM was used
        summary: Human-readable summary
        claims: List of claims with evidence
        artifacts: List of produced artifacts
        status: Execution status
        warnings: Any warnings
        errors: Any errors
        next_actions: Suggested follow-up actions

    Returns:
        Validated ModuleOutput object
    """
    return ModuleOutput(
        module=module,
        model_used=model_used,
        summary=summary,
        status=status,
        claims=[Claim.model_validate(c) for c in (claims or [])],
        artifacts=[Artifact.model_validate(a) for a in (artifacts or [])],
        warnings=warnings or [],
        errors=[ErrorInfo.model_validate(e) for e in (errors or [])],
        next_actions=[NextAction.model_validate(n) for n in (next_actions or [])],
    )
