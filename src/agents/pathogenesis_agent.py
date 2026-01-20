"""Pathogenesis Discovery Agent for AgingResearchAI.

This agent orchestrates the discovery of disease mechanisms by:
1. Searching for relevant datasets (Lobster/GEO)
2. Running omics analysis (Lobster)
3. Retrieving literature (Gemini/RAG)
4. Synthesizing findings (Claude)

Following the key principle: LLMs write code, tools execute, LLMs interpret.
"""

import json
from pathlib import Path
from typing import Any

from ..core.router import ModelRouter, TaskType
from ..core.schema import (
    EvidenceTier,
    ModelType,
    ModuleOutput,
    ModuleStatus,
    create_output,
)
from ..models.claude_client import ClaudeClient
from ..models.gemini_client import GeminiClient
from ..models.lobster_client import LobsterClient


class PathogenesisAgent:
    """Agent for discovering disease pathogenesis mechanisms.

    Workflow:
    1. Dataset Discovery (Lobster) - Find relevant GEO datasets
    2. Omics Analysis (Lobster) - QC, DE, pathway enrichment
    3. Literature RAG (Gemini) - Retrieve relevant papers
    4. Synthesis (Claude) - Integrate and propose mechanisms

    Usage:
        agent = PathogenesisAgent()
        result = agent.discover("Type 2 Diabetes", tissue="liver")
    """

    def __init__(
        self,
        gemini: GeminiClient | None = None,
        claude: ClaudeClient | None = None,
        lobster: LobsterClient | None = None,
        router: ModelRouter | None = None,
    ):
        """Initialize pathogenesis agent.

        Args:
            gemini: Gemini client (creates default if None)
            claude: Claude client (creates default if None)
            lobster: Lobster client (creates default if None)
            router: Model router (creates default if None)
        """
        self.router = router or ModelRouter()

        # Initialize clients lazily to allow partial initialization
        self._gemini = gemini
        self._claude = claude
        self._lobster = lobster

    @property
    def gemini(self) -> GeminiClient:
        if self._gemini is None:
            self._gemini = GeminiClient()
        return self._gemini

    @property
    def claude(self) -> ClaudeClient:
        if self._claude is None:
            self._claude = ClaudeClient()
        return self._claude

    @property
    def lobster(self) -> LobsterClient:
        if self._lobster is None:
            self._lobster = LobsterClient()
        return self._lobster

    def discover(
        self,
        disease: str,
        tissue: str | None = None,
        organism: str = "human",
        existing_datasets: list[str] | None = None,
    ) -> ModuleOutput:
        """Discover pathogenesis mechanisms for a disease.

        Args:
            disease: Disease name (e.g., "Type 2 Diabetes")
            tissue: Target tissue (e.g., "liver", "muscle")
            organism: Organism (default: human)
            existing_datasets: Optional list of GEO accessions to use

        Returns:
            ModuleOutput with discovered mechanisms
        """
        all_claims = []
        all_artifacts = []
        warnings = []

        # Step 1: Dataset Discovery
        if existing_datasets:
            dataset_info = {"accessions": existing_datasets}
        else:
            dataset_result = self._discover_datasets(disease, tissue, organism)
            if dataset_result.status == ModuleStatus.FAILED:
                warnings.append(f"Dataset discovery failed: {dataset_result.summary}")
                dataset_info = {"accessions": []}
            else:
                all_claims.extend(dataset_result.claims)
                all_artifacts.extend(dataset_result.artifacts)
                dataset_info = self._extract_datasets(dataset_result)

        # Step 2: Omics Analysis (if datasets found)
        omics_data = {}
        if dataset_info.get("accessions"):
            omics_result = self._analyze_omics(dataset_info["accessions"][0])
            if omics_result.status == ModuleStatus.FAILED:
                warnings.append(f"Omics analysis failed: {omics_result.summary}")
            else:
                all_claims.extend(omics_result.claims)
                all_artifacts.extend(omics_result.artifacts)
                omics_data = self._extract_omics_data(omics_result)
        else:
            warnings.append("No datasets found for omics analysis")

        # Step 3: Literature Retrieval
        lit_result = self._retrieve_literature(disease, tissue, omics_data)
        all_claims.extend(lit_result.claims)
        literature_context = self._extract_literature(lit_result)

        # Step 4: Synthesis (Claude)
        synthesis_result = self._synthesize(
            disease=disease,
            tissue=tissue,
            omics_data=omics_data,
            literature_context=literature_context,
        )

        # Combine all results
        final_claims = all_claims + synthesis_result.claims
        final_artifacts = all_artifacts + synthesis_result.artifacts

        return create_output(
            module="pathogenesis_discovery",
            model_used=ModelType.CLAUDE,  # Final synthesis uses Claude
            summary=synthesis_result.summary,
            status=synthesis_result.status,
            claims=[c.model_dump() for c in final_claims] if final_claims else [],
            artifacts=[a.model_dump() for a in final_artifacts] if final_artifacts else [],
            warnings=warnings + synthesis_result.warnings,
            next_actions=[na.model_dump() for na in synthesis_result.next_actions],
        )

    def _discover_datasets(
        self,
        disease: str,
        tissue: str | None,
        organism: str,
    ) -> ModuleOutput:
        """Search GEO for relevant datasets."""
        query_parts = [disease, organism]
        if tissue:
            query_parts.append(tissue)
        query = " ".join(query_parts)

        return self.lobster.search_geo(query, data_type="RNA-seq")

    def _analyze_omics(self, accession: str) -> ModuleOutput:
        """Run omics analysis on a dataset."""
        # Download and QC
        download_result = self.lobster.download_dataset(accession)
        if download_result.status == ModuleStatus.FAILED:
            return download_result

        # For now, return the download result
        # In full implementation, would chain DE and pathway analysis
        return download_result

    def _retrieve_literature(
        self,
        disease: str,
        tissue: str | None,
        omics_data: dict[str, Any],
    ) -> ModuleOutput:
        """Retrieve relevant literature using Gemini + Lobster."""
        # Build query from disease and top genes if available
        query_parts = [disease, "pathogenesis", "mechanism"]
        if tissue:
            query_parts.append(tissue)

        # Add top differentially expressed genes if available
        top_genes = omics_data.get("top_genes", [])[:5]
        if top_genes:
            query_parts.extend(top_genes)

        query = " ".join(query_parts)

        # Use Lobster for PubMed search
        return self.lobster.search_literature(query)

    def _synthesize(
        self,
        disease: str,
        tissue: str | None,
        omics_data: dict[str, Any],
        literature_context: str,
    ) -> ModuleOutput:
        """Synthesize findings using Claude."""
        return self.claude.synthesize_pathogenesis(
            disease=disease,
            omics_data=omics_data,
            literature_context=literature_context,
        )

    def _extract_datasets(self, result: ModuleOutput) -> dict[str, Any]:
        """Extract dataset information from search result."""
        # In real implementation, would parse Lobster output
        return {"accessions": []}

    def _extract_omics_data(self, result: ModuleOutput) -> dict[str, Any]:
        """Extract omics data from analysis result."""
        return {
            "top_genes": [],
            "pathways": [],
            "de_summary": result.summary,
        }

    def _extract_literature(self, result: ModuleOutput) -> str:
        """Extract literature context for synthesis."""
        return result.summary


class QuickPathogenesisCheck:
    """Quick pathogenesis check using only Gemini (cost-effective).

    For initial hypothesis generation before full analysis.
    """

    def __init__(self, gemini: GeminiClient | None = None):
        self.gemini = gemini or GeminiClient()

    def quick_check(
        self,
        disease: str,
        hypothesis: str | None = None,
    ) -> ModuleOutput:
        """Quick literature-based pathogenesis check.

        Args:
            disease: Disease name
            hypothesis: Optional specific hypothesis to evaluate

        Returns:
            ModuleOutput with initial assessment
        """
        prompt = f"""Based on your knowledge, provide a brief overview of known pathogenesis for {disease}.

{f"Specifically evaluate this hypothesis: {hypothesis}" if hypothesis else ""}

Format response as JSON:
{{
    "known_mechanisms": ["mechanism 1", "mechanism 2"],
    "key_pathways": ["pathway 1"],
    "key_genes": ["gene1", "gene2"],
    "evidence_strength": "strong|moderate|weak",
    "knowledge_gaps": ["gap 1"],
    "suggested_analyses": ["analysis 1"]
}}"""

        try:
            result = self.gemini.generate_json(prompt)

            return create_output(
                module="quick_pathogenesis_check",
                model_used=ModelType.GEMINI,
                summary=f"Quick check for {disease} pathogenesis",
                claims=[
                    {
                        "text": f"Identified {len(result.get('known_mechanisms', []))} known mechanisms",
                        "confidence": 0.6,  # Lower confidence for quick check
                        "evidence_tier": "tier3",  # Needs validation
                        "evidence": [
                            {"type": "computed", "tool": "gemini", "artifact_id": "quick_check"}
                        ],
                    }
                ],
                warnings=["This is a quick check based on model knowledge. Validate with literature search."],
            )

        except Exception as e:
            return create_output(
                module="quick_pathogenesis_check",
                model_used=ModelType.GEMINI,
                summary=f"Quick check failed: {e}",
                status=ModuleStatus.FAILED,
                errors=[{"code": "QUICK_CHECK_FAILED", "message": str(e)}],
            )
