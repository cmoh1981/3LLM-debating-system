"""Evidence tier classification for AgingResearchAI.

Based on LongevityBench principles:
- Tier 1: Replicated + Causal + Literature (highest confidence)
- Tier 2: Replicated association
- Tier 3: Single analysis only (needs validation)
"""

from dataclasses import dataclass
from enum import Enum

from .schema import Claim, Evidence, EvidenceTier, EvidenceType


class ReplicationStatus(str, Enum):
    """Replication status of a finding."""

    REPLICATED = "replicated"  # Multiple independent studies
    SINGLE = "single"  # Only one study/analysis
    CONFLICTING = "conflicting"  # Mixed results


class CausalSupport(str, Enum):
    """Level of causal evidence."""

    CAUSAL = "causal"  # Mechanistic/intervention evidence
    ASSOCIATIVE = "associative"  # Correlation only
    UNKNOWN = "unknown"


@dataclass
class EvidenceAssessment:
    """Assessment of evidence quality for a claim."""

    replication: ReplicationStatus
    causal_support: CausalSupport
    has_literature: bool
    has_database: bool
    has_computed: bool
    pmid_count: int
    tier: EvidenceTier
    explanation: str


class EvidenceClassifier:
    """Classifies evidence into tiers based on quality criteria.

    Tier 1 criteria:
    - Replicated across multiple studies
    - Causal support (mechanistic or intervention data)
    - Literature evidence with citations

    Tier 2 criteria:
    - Replicated association
    - May lack causal support

    Tier 3 criteria:
    - Single analysis only
    - Needs validation before action
    """

    def __init__(
        self,
        min_pmids_for_replication: int = 2,
        require_causal_for_tier1: bool = True,
    ):
        """Initialize classifier with thresholds.

        Args:
            min_pmids_for_replication: Minimum PMIDs to consider "replicated"
            require_causal_for_tier1: Whether causal support needed for Tier 1
        """
        self.min_pmids_for_replication = min_pmids_for_replication
        self.require_causal_for_tier1 = require_causal_for_tier1

    def classify(
        self,
        evidence_list: list[Evidence],
        replication_status: ReplicationStatus | None = None,
        causal_support: CausalSupport | None = None,
    ) -> EvidenceAssessment:
        """Classify evidence into a tier.

        Args:
            evidence_list: List of evidence entries
            replication_status: Override for replication assessment
            causal_support: Override for causal assessment

        Returns:
            EvidenceAssessment with tier and explanation
        """
        # Count evidence types
        pmids = set()
        has_database = False
        has_computed = False

        for e in evidence_list:
            if e.type == EvidenceType.LITERATURE and e.pmid:
                pmids.add(e.pmid)
            elif e.type == EvidenceType.DATABASE:
                has_database = True
            elif e.type == EvidenceType.COMPUTED:
                has_computed = True

        pmid_count = len(pmids)
        has_literature = pmid_count > 0

        # Determine replication status if not provided
        if replication_status is None:
            if pmid_count >= self.min_pmids_for_replication:
                replication_status = ReplicationStatus.REPLICATED
            else:
                replication_status = ReplicationStatus.SINGLE

        # Default causal support if not provided
        if causal_support is None:
            causal_support = CausalSupport.UNKNOWN

        # Classify tier
        tier, explanation = self._determine_tier(
            replication_status,
            causal_support,
            has_literature,
            pmid_count,
        )

        return EvidenceAssessment(
            replication=replication_status,
            causal_support=causal_support,
            has_literature=has_literature,
            has_database=has_database,
            has_computed=has_computed,
            pmid_count=pmid_count,
            tier=tier,
            explanation=explanation,
        )

    def _determine_tier(
        self,
        replication: ReplicationStatus,
        causal: CausalSupport,
        has_lit: bool,
        pmid_count: int,
    ) -> tuple[EvidenceTier, str]:
        """Determine tier based on criteria."""
        # Tier 1: Replicated + Causal + Literature
        if (
            replication == ReplicationStatus.REPLICATED
            and has_lit
            and (causal == CausalSupport.CAUSAL or not self.require_causal_for_tier1)
        ):
            return (
                EvidenceTier.TIER1,
                f"Tier 1: Replicated ({pmid_count} studies), "
                f"causal support={causal.value}, literature evidence present",
            )

        # Tier 2: Replicated but may lack causal
        if replication == ReplicationStatus.REPLICATED:
            return (
                EvidenceTier.TIER2,
                f"Tier 2: Replicated ({pmid_count} studies), "
                f"but causal support={causal.value}",
            )

        # Tier 3: Single analysis
        return (
            EvidenceTier.TIER3,
            f"Tier 3: Single analysis only ({pmid_count} PMIDs). "
            "Needs validation before action.",
        )

    def upgrade_tier(
        self,
        current_tier: EvidenceTier,
        new_evidence: list[Evidence],
    ) -> EvidenceTier:
        """Check if new evidence upgrades the tier.

        Args:
            current_tier: Current evidence tier
            new_evidence: Additional evidence to consider

        Returns:
            New tier (same or higher)
        """
        # Can't upgrade from Tier 1
        if current_tier == EvidenceTier.TIER1:
            return EvidenceTier.TIER1

        assessment = self.classify(new_evidence)

        # Return higher tier
        tier_order = {EvidenceTier.TIER1: 3, EvidenceTier.TIER2: 2, EvidenceTier.TIER3: 1}
        if tier_order[assessment.tier] > tier_order[current_tier]:
            return assessment.tier
        return current_tier

    def validate_claim(self, claim: Claim) -> tuple[bool, str]:
        """Validate that a claim meets minimum evidence standards.

        Args:
            claim: Claim to validate

        Returns:
            Tuple of (is_valid, reason)
        """
        if not claim.evidence:
            return False, "Claim has no evidence (violates no-uncited-claims rule)"

        if claim.confidence > 0.8 and claim.evidence_tier == EvidenceTier.TIER3:
            return (
                False,
                "High confidence (>0.8) with Tier 3 evidence is not allowed",
            )

        # Check for at least one valid evidence type
        valid_evidence = any(
            (e.type == EvidenceType.LITERATURE and e.pmid)
            or (e.type == EvidenceType.DATABASE and e.source)
            or (e.type == EvidenceType.COMPUTED and e.artifact_id)
            for e in claim.evidence
        )

        if not valid_evidence:
            return False, "Evidence entries are incomplete"

        return True, "Claim meets evidence standards"
