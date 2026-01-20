"""Debate API routes for AgingResearchAI.

Endpoints for the multi-LLM debate system.
"""

import logging
import time

from fastapi import APIRouter, HTTPException

from ..dependencies import get_debate_engine, get_available_debate_clients
from ..models import (
    DebateRequest,
    DebateResponse,
    DebateClaimResponse,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/",
    response_model=DebateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Debate service unavailable"},
    },
    summary="Start a debate session",
    description="""
Run a multi-LLM debate on a scientific topic.

The debate follows this workflow:
1. **Proposal Phase**: Gemini proposes claims (FREE)
2. **Critique Phase**: DeepSeek critiques claims (~$0.14/M tokens)
3. **Support Phase**: Kimi provides third perspective (~$0.20/M tokens)
4. **Voting Phase**: All models vote, 2/3 majority for consensus

Returns consensus claims, rejected claims, and unresolved claims.
    """,
)
async def start_debate(request: DebateRequest):
    """Run multi-LLM debate on a scientific topic."""
    start_time = time.time()

    # Get debate engine
    config = {
        "max_rounds": request.max_rounds,
        "consensus_threshold": request.consensus_threshold,
    }
    engine = get_debate_engine(config)

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Debate service unavailable. Need at least 2 LLM clients configured.",
        )

    # Format initial claims
    initial_claims = [
        {
            "text": c.text,
            "proposer": c.proposer,
            "evidence": c.evidence,
        }
        for c in request.claims
    ]

    try:
        # Run debate
        result = engine.debate(
            topic=request.topic,
            initial_claims=initial_claims if initial_claims else None,
            context=request.context,
        )

        # Format response
        elapsed_ms = int((time.time() - start_time) * 1000)

        return DebateResponse(
            topic=result.topic,
            rounds=result.rounds,
            consensus_claims=[
                DebateClaimResponse(
                    text=c["text"],
                    proposer=c["proposer"],
                    confidence=c["confidence"],
                    votes=c["votes"],
                    critiques_count=c["critiques_count"],
                )
                for c in result.consensus_claims
            ],
            rejected_claims=[
                DebateClaimResponse(
                    text=c["text"],
                    proposer=c["proposer"],
                    confidence=c["confidence"],
                    votes=c["votes"],
                    critiques_count=c["critiques_count"],
                )
                for c in result.rejected_claims
            ],
            unresolved_claims=[
                DebateClaimResponse(
                    text=c["text"],
                    proposer=c["proposer"],
                    confidence=c["confidence"],
                    votes=c["votes"],
                    critiques_count=c["critiques_count"],
                )
                for c in result.unresolved_claims
            ],
            overall_confidence=result.overall_confidence,
            debate_summary=result.debate_summary,
            elapsed_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.exception(f"Debate failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Debate failed: {str(e)}",
        )


@router.get(
    "/status",
    summary="Get debate system status",
    description="Check which LLM clients are available for debate.",
)
async def debate_status():
    """Get debate system status."""
    clients = get_available_debate_clients()

    # Determine roles
    roles = {}
    if "gemini" in clients:
        roles["proposer"] = "gemini"
    if "deepseek" in clients:
        roles["critic"] = "deepseek"
    elif "claude" in clients:
        roles["critic"] = "claude"
    if "kimi" in clients:
        roles["judge"] = "kimi"
    elif "openai" in clients:
        roles["judge"] = "openai"

    return {
        "available_clients": list(clients.keys()),
        "client_count": len(clients),
        "debate_ready": len(clients) >= 2,
        "assigned_roles": roles,
        "cost_estimate": _estimate_debate_cost(roles),
    }


def _estimate_debate_cost(roles: dict) -> dict:
    """Estimate cost per debate based on assigned roles."""
    # Approximate costs per 1M tokens
    costs = {
        "gemini": 0.0,  # Free tier
        "deepseek": 0.14,
        "kimi": 0.20,
        "claude": 3.0,  # Claude Sonnet
        "openai": 2.5,  # GPT-4o
    }

    # Assume ~10K tokens per debate
    tokens_per_debate = 10000 / 1_000_000

    estimated_cost = 0.0
    for role, model in roles.items():
        estimated_cost += costs.get(model, 0) * tokens_per_debate

    return {
        "per_debate_usd": round(estimated_cost, 4),
        "daily_estimate_100_debates": round(estimated_cost * 100, 2),
    }


@router.post(
    "/quick",
    response_model=DebateResponse,
    summary="Quick single-round debate",
    description="Run a simplified single-round debate for faster results.",
)
async def quick_debate(request: DebateRequest):
    """Run a quick single-round debate."""
    # Override to single round
    request.max_rounds = 1
    return await start_debate(request)
