"""Pipeline API routes for AgingResearchAI.

Endpoints for running full research pipelines.
"""

import logging
import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks

from ..dependencies import (
    get_debate_engine,
    get_knowledge_base,
    get_retriever,
    get_gemini_client,
)
from ..models import (
    PipelineRequest,
    PipelineResponse,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for pipeline runs (use database in production)
_pipeline_runs: dict[str, dict] = {}


@router.post(
    "/run",
    response_model=PipelineResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Pipeline service unavailable"},
    },
    summary="Run research pipeline",
    description="""
Run a complete research pipeline for drug discovery.

Available workflows:
- **pathogenesis**: Discover disease pathogenesis mechanisms
- **target_discovery**: Identify potential drug targets
- **full**: Complete end-to-end pipeline

The pipeline:
1. Retrieves relevant literature from RAG (if enabled)
2. Generates scientific claims using LLMs
3. Validates claims through multi-LLM debate (if enabled)
4. Returns verified findings with evidence
    """,
)
async def run_pipeline(request: PipelineRequest):
    """Run a research pipeline."""
    start_time = time.time()
    run_id = str(uuid4())[:8]

    # Validate disease
    valid_diseases = ["T2D", "Type 2 Diabetes", "NAFLD", "NASH", "Sarcopenia"]
    if not any(d.lower() in request.disease.lower() for d in valid_diseases):
        logger.warning(f"Non-standard disease requested: {request.disease}")

    try:
        # Step 1: RAG context retrieval
        context = ""
        if request.use_rag:
            retriever = get_retriever("literature")
            if retriever:
                query = f"{request.disease} {request.tissue} pathogenesis mechanisms"
                results = retriever.retrieve(query, top_k=10)
                context = "\n\n".join([
                    f"[{i+1}] {r.document.content[:500]}..."
                    for i, r in enumerate(results)
                ])
                logger.info(f"Retrieved {len(results)} documents for context")

        # Step 2: Generate claims using LLM
        claims = []
        gemini = get_gemini_client()

        if gemini:
            prompt = _build_research_prompt(
                disease=request.disease,
                tissue=request.tissue,
                workflow=request.workflow_type,
                context=context,
            )

            try:
                response = gemini.generate(prompt)
                # Parse claims from response
                claims = _parse_claims(response)
            except Exception as e:
                logger.warning(f"Claim generation failed: {e}")

        # Step 3: Debate validation (if enabled)
        debate_results = None
        if request.enable_debate and claims:
            engine = get_debate_engine()
            if engine:
                initial_claims = [
                    {"text": c["text"], "proposer": "gemini", "evidence": c.get("evidence", [])}
                    for c in claims
                ]

                topic = f"{request.disease} {request.workflow_type} in {request.tissue}"
                result = engine.debate(
                    topic=topic,
                    initial_claims=initial_claims,
                    context=context,
                )

                debate_results = {
                    "topic": result.topic,
                    "rounds": result.rounds,
                    "consensus_count": len(result.consensus_claims),
                    "rejected_count": len(result.rejected_claims),
                    "unresolved_count": len(result.unresolved_claims),
                    "overall_confidence": result.overall_confidence,
                    "summary": result.debate_summary,
                }

                # Update claims with debate results
                claims = _merge_debate_results(claims, result)

        # Step 4: Generate next actions
        next_actions = _generate_next_actions(
            disease=request.disease,
            workflow=request.workflow_type,
            claims=claims,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        response = PipelineResponse(
            run_id=run_id,
            disease=request.disease,
            tissue=request.tissue,
            workflow_type=request.workflow_type,
            status="completed",
            summary=_generate_summary(request.disease, claims, debate_results),
            claims=claims,
            artifacts=[],  # Would contain generated figures, tables, etc.
            debate_results=debate_results,
            next_actions=next_actions,
            elapsed_time_ms=elapsed_ms,
        )

        # Store run for later retrieval
        _pipeline_runs[run_id] = response.model_dump()

        return response

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(e)}",
        )


@router.get(
    "/runs/{run_id}",
    summary="Get pipeline run",
    description="Retrieve results of a previous pipeline run.",
)
async def get_run(run_id: str):
    """Get a previous pipeline run."""
    if run_id not in _pipeline_runs:
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id} not found",
        )

    return _pipeline_runs[run_id]


@router.get(
    "/runs",
    summary="List pipeline runs",
    description="List recent pipeline runs.",
)
async def list_runs(limit: int = 10):
    """List recent pipeline runs."""
    runs = list(_pipeline_runs.values())[-limit:]
    return {
        "total": len(_pipeline_runs),
        "runs": [
            {
                "run_id": r["run_id"],
                "disease": r["disease"],
                "workflow_type": r["workflow_type"],
                "status": r["status"],
            }
            for r in runs
        ],
    }


@router.get(
    "/workflows",
    summary="List available workflows",
    description="List available research workflows.",
)
async def list_workflows():
    """List available workflows."""
    return {
        "workflows": [
            {
                "id": "pathogenesis",
                "name": "Pathogenesis Discovery",
                "description": "Discover disease mechanisms and pathways",
            },
            {
                "id": "target_discovery",
                "name": "Target Discovery",
                "description": "Identify potential drug targets",
            },
            {
                "id": "full",
                "name": "Full Pipeline",
                "description": "Complete end-to-end analysis",
            },
        ],
        "diseases": [
            {
                "id": "T2D",
                "name": "Type 2 Diabetes",
                "aliases": ["Type 2 Diabetes", "T2DM"],
            },
            {
                "id": "NAFLD",
                "name": "NAFLD/NASH",
                "aliases": ["NAFLD", "NASH", "Fatty Liver"],
            },
            {
                "id": "Sarcopenia",
                "name": "Sarcopenia",
                "aliases": ["Muscle Wasting"],
            },
        ],
        "tissues": ["liver", "muscle", "adipose", "pancreas", "brain"],
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _build_research_prompt(
    disease: str,
    tissue: str,
    workflow: str,
    context: str,
) -> str:
    """Build research prompt for LLM."""
    base_prompt = f"""You are a biomedical research AI analyzing {disease} in {tissue} tissue.

Workflow: {workflow}

Context from literature:
{context[:3000] if context else "No additional context available."}

Based on the context and your knowledge, identify key scientific claims about:
"""

    if workflow == "pathogenesis":
        base_prompt += """
- Disease mechanisms and pathways
- Key genes and proteins involved
- Regulatory networks
- Potential therapeutic targets

For each claim, provide:
1. The claim statement
2. Supporting evidence (PMIDs if available)
3. Confidence level (high/medium/low)
"""
    elif workflow == "target_discovery":
        base_prompt += """
- Druggable targets
- Target validation evidence
- Known modulators/inhibitors
- Safety considerations

For each target, provide:
1. Target name and type
2. Evidence for disease relevance
3. Druggability assessment
4. Known compounds that modulate it
"""
    else:
        base_prompt += """
- Disease mechanisms
- Drug targets
- ADMET considerations
- Next experimental steps
"""

    base_prompt += """

Format your response as a list of claims with evidence.
Be specific and cite sources when possible.
"""

    return base_prompt


def _parse_claims(response: str) -> list[dict]:
    """Parse claims from LLM response."""
    # Simple parsing - in production, use structured output
    claims = []

    lines = response.split("\n")
    current_claim = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for numbered claims or bullet points
        if line[0].isdigit() or line.startswith("-") or line.startswith("*"):
            if current_claim:
                claims.append(current_claim)

            text = line.lstrip("0123456789.-*) ")
            current_claim = {
                "text": text,
                "confidence": 0.7,
                "evidence": [],
                "status": "proposed",
            }
        elif current_claim:
            # Look for PMIDs
            if "PMID" in line or "pmid" in line.lower():
                current_claim["evidence"].append(line)

    if current_claim:
        claims.append(current_claim)

    return claims[:10]  # Limit to 10 claims


def _merge_debate_results(claims: list[dict], debate_result) -> list[dict]:
    """Merge debate results into claims."""
    # Create lookup by claim text
    consensus_texts = {c["text"] for c in debate_result.consensus_claims}
    rejected_texts = {c["text"] for c in debate_result.rejected_claims}

    for claim in claims:
        if claim["text"] in consensus_texts:
            claim["status"] = "consensus"
            # Find matching consensus claim for confidence
            for dc in debate_result.consensus_claims:
                if dc["text"] == claim["text"]:
                    claim["confidence"] = dc["confidence"]
                    break
        elif claim["text"] in rejected_texts:
            claim["status"] = "rejected"
            claim["confidence"] = 0.0
        else:
            claim["status"] = "unresolved"

    return claims


def _generate_next_actions(
    disease: str,
    workflow: str,
    claims: list[dict],
) -> list[dict]:
    """Generate suggested next actions."""
    actions = []

    # Count consensus claims
    consensus_count = sum(1 for c in claims if c.get("status") == "consensus")

    if consensus_count > 0:
        actions.append({
            "action": "Validate top targets experimentally",
            "priority": "P0",
            "reason": f"{consensus_count} claims reached consensus",
        })

    if workflow == "pathogenesis":
        actions.append({
            "action": "Run pathway enrichment analysis",
            "priority": "P1",
            "reason": "Identify key signaling pathways",
        })

    actions.append({
        "action": "Search for compound modulators",
        "priority": "P1",
        "reason": "Identify existing tool compounds",
    })

    actions.append({
        "action": "Run ADMET predictions on candidate compounds",
        "priority": "P2",
        "reason": "Assess drug-likeness early",
    })

    return actions


def _generate_summary(
    disease: str,
    claims: list[dict],
    debate_results: dict | None,
) -> str:
    """Generate pipeline summary."""
    summary_parts = [f"Research pipeline completed for {disease}."]

    if claims:
        summary_parts.append(f"Generated {len(claims)} scientific claims.")

    if debate_results:
        summary_parts.append(
            f"Debate validation: {debate_results['consensus_count']} consensus, "
            f"{debate_results['rejected_count']} rejected, "
            f"{debate_results['unresolved_count']} unresolved."
        )
        if debate_results['overall_confidence'] > 0:
            summary_parts.append(
                f"Overall confidence: {debate_results['overall_confidence']:.0%}"
            )

    return " ".join(summary_parts)
