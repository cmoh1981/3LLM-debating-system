"""Tests for core schema validation."""

import pytest
from pydantic import ValidationError

from src.core.schema import (
    Claim,
    Evidence,
    EvidenceTier,
    EvidenceType,
    ModuleOutput,
    ModuleStatus,
    ModelType,
    create_output,
    validate_output,
)


class TestEvidence:
    """Test Evidence model."""

    def test_valid_literature_evidence(self):
        """Test valid literature evidence with PMID."""
        evidence = Evidence(
            type=EvidenceType.LITERATURE,
            pmid="12345678",
            quote="This study demonstrates...",
        )
        assert evidence.pmid == "12345678"

    def test_valid_database_evidence(self):
        """Test valid database evidence."""
        evidence = Evidence(
            type=EvidenceType.DATABASE,
            source="UniProt",
            id="P04637",
        )
        assert evidence.source == "UniProt"

    def test_valid_computed_evidence(self):
        """Test valid computed evidence."""
        evidence = Evidence(
            type=EvidenceType.COMPUTED,
            tool="deseq2",
            artifact_id="de_results_001",
        )
        assert evidence.tool == "deseq2"


class TestClaim:
    """Test Claim model."""

    def test_valid_claim(self):
        """Test valid claim with evidence."""
        claim = Claim(
            text="AMPK signaling is downregulated in T2D liver samples",
            confidence=0.85,
            evidence_tier=EvidenceTier.TIER1,
            evidence=[
                Evidence(
                    type=EvidenceType.LITERATURE,
                    pmid="12345678",
                    quote="AMPK activity reduced...",
                )
            ],
        )
        assert claim.confidence == 0.85

    def test_claim_requires_evidence(self):
        """Test that claims require at least one evidence entry."""
        with pytest.raises(ValidationError):
            Claim(
                text="This claim has no evidence",
                confidence=0.5,
                evidence_tier=EvidenceTier.TIER3,
                evidence=[],  # Empty evidence should fail
            )

    def test_claim_text_minimum_length(self):
        """Test that claim text has minimum length."""
        with pytest.raises(ValidationError):
            Claim(
                text="Short",  # Too short
                confidence=0.5,
                evidence_tier=EvidenceTier.TIER3,
                evidence=[
                    Evidence(type=EvidenceType.COMPUTED, tool="test", artifact_id="test")
                ],
            )


class TestModuleOutput:
    """Test ModuleOutput model."""

    def test_valid_module_output(self):
        """Test valid module output."""
        output = ModuleOutput(
            module="pathogenesis",
            model_used=ModelType.CLAUDE,
            status=ModuleStatus.OK,
            summary="Identified 3 key pathways in T2D pathogenesis",
            claims=[
                Claim(
                    text="AMPK signaling is downregulated in T2D liver samples",
                    confidence=0.85,
                    evidence_tier=EvidenceTier.TIER1,
                    evidence=[
                        Evidence(
                            type=EvidenceType.LITERATURE,
                            pmid="12345678",
                            quote="AMPK activity reduced...",
                        )
                    ],
                )
            ],
        )
        assert output.module == "pathogenesis"
        assert len(output.claims) == 1

    def test_output_has_run_id(self):
        """Test that output automatically gets run_id."""
        output = ModuleOutput(
            module="test",
            model_used=ModelType.GEMINI,
            status=ModuleStatus.OK,
            summary="Test summary with enough characters",
        )
        assert output.run_id is not None

    def test_output_has_timestamp(self):
        """Test that output automatically gets timestamp."""
        output = ModuleOutput(
            module="test",
            model_used=ModelType.GEMINI,
            status=ModuleStatus.OK,
            summary="Test summary with enough characters",
        )
        assert output.timestamp is not None


class TestCreateOutput:
    """Test create_output helper."""

    def test_create_output_basic(self):
        """Test basic output creation."""
        output = create_output(
            module="test_module",
            model_used=ModelType.GEMINI,
            summary="Test summary with enough length",
        )
        assert output.module == "test_module"
        assert output.status == ModuleStatus.OK

    def test_create_output_with_claims(self):
        """Test output creation with claims."""
        output = create_output(
            module="test_module",
            model_used=ModelType.CLAUDE,
            summary="Test with claims and enough length",
            claims=[
                {
                    "text": "This is a test claim with sufficient length",
                    "confidence": 0.7,
                    "evidence_tier": "tier2",
                    "evidence": [
                        {"type": "computed", "tool": "test", "artifact_id": "test_001"}
                    ],
                }
            ],
        )
        assert len(output.claims) == 1
        assert output.claims[0].confidence == 0.7


class TestValidateOutput:
    """Test validate_output function."""

    def test_validate_valid_data(self):
        """Test validation of valid data."""
        data = {
            "run_id": "test-123",
            "module": "pathogenesis",
            "model_used": "claude",
            "status": "ok",
            "timestamp": "2026-01-20T10:00:00",
            "summary": "Test summary with sufficient length",
            "claims": [],
            "artifacts": [],
            "next_actions": [],
            "warnings": [],
            "errors": [],
        }
        output = validate_output(data)
        assert output.module == "pathogenesis"

    def test_validate_invalid_data(self):
        """Test validation rejects invalid data."""
        data = {
            "module": "test",
            # Missing required fields
        }
        with pytest.raises(ValidationError):
            validate_output(data)
