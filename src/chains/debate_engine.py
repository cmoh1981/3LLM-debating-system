"""Multi-LLM Debate Engine for AgingResearchAI.

Implements a scientific debate system where multiple LLMs:
1. Propose claims independently
2. Critique each other's claims
3. Vote on claim validity
4. Reach consensus through structured debate

This improves scientific reliability through:
- Error detection via adversarial review
- Diverse reasoning perspectives
- Evidence-based consensus building

Based on research showing multi-agent debate improves LLM accuracy.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class DebateRole(str, Enum):
    """Roles in the debate."""
    PROPOSER = "proposer"      # Makes initial claim
    CRITIC = "critic"          # Challenges claims
    SUPPORTER = "supporter"    # Provides supporting evidence
    JUDGE = "judge"            # Final arbitration


class VoteType(str, Enum):
    """Vote types for claims."""
    SUPPORT = "support"
    OPPOSE = "oppose"
    ABSTAIN = "abstain"
    NEEDS_EVIDENCE = "needs_evidence"


class ClaimStatus(str, Enum):
    """Status of a debated claim."""
    PROPOSED = "proposed"
    UNDER_DEBATE = "under_debate"
    CONSENSUS_REACHED = "consensus_reached"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


@dataclass
class DebateClaim:
    """A claim being debated."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    text: str = ""
    proposer: str = ""  # Model name
    evidence: list[dict] = field(default_factory=list)
    critiques: list[dict] = field(default_factory=list)
    supports: list[dict] = field(default_factory=list)
    votes: dict[str, VoteType] = field(default_factory=dict)
    status: ClaimStatus = ClaimStatus.PROPOSED
    final_confidence: float = 0.0
    revision_history: list[str] = field(default_factory=list)


@dataclass
class DebateRound:
    """A single round of debate."""
    round_number: int
    topic: str
    claims: list[DebateClaim] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    summary: str = ""


class DebateConfig(BaseModel):
    """Configuration for debate engine."""

    max_rounds: int = 3
    consensus_threshold: float = 0.66  # 2/3 majority
    min_evidence_per_claim: int = 1
    enable_revision: bool = True
    require_citations: bool = True

    # Model roles assignment
    # Gemini proposes (fast, free)
    # Grok critiques (reasoning, xAI)
    # DeepSeek provides third vote (independent, cheap)
    default_proposer: str = "gemini"
    default_critic: str = "grok"
    default_judge: str = "deepseek"


class DebateResult(BaseModel):
    """Result of a debate session."""

    topic: str
    rounds: int
    consensus_claims: list[dict] = Field(default_factory=list)
    rejected_claims: list[dict] = Field(default_factory=list)
    unresolved_claims: list[dict] = Field(default_factory=list)
    overall_confidence: float = 0.0
    debate_summary: str = ""


# =============================================================================
# Debate Engine
# =============================================================================

class DebateEngine:
    """Multi-LLM debate engine for scientific claim verification.

    Orchestrates structured debates between multiple LLMs to:
    - Cross-verify scientific claims
    - Identify errors and hallucinations
    - Build consensus on findings
    - Improve evidence quality

    Usage:
        engine = DebateEngine(clients={"gemini": gemini, "claude": claude, "qwen": qwen})
        result = engine.debate(
            topic="AMPK's role in T2D pathogenesis",
            initial_claims=[...],
        )
    """

    def __init__(
        self,
        clients: dict[str, Any] | None = None,
        config: DebateConfig | None = None,
    ):
        """Initialize debate engine.

        Args:
            clients: Dict mapping model names to client instances
            config: Debate configuration
        """
        self.clients = clients or {}
        self.config = config or DebateConfig()
        self.debate_history: list[DebateRound] = []

    def add_client(self, name: str, client: Any):
        """Add a model client.

        Args:
            name: Model identifier (gemini, claude, qwen)
            client: Client instance with generate() method
        """
        self.clients[name] = client
        logger.info(f"Added debate participant: {name}")

    def debate(
        self,
        topic: str,
        initial_claims: list[dict] | None = None,
        context: str = "",
    ) -> DebateResult:
        """Run a full debate session.

        Args:
            topic: Topic/question to debate
            initial_claims: Optional starting claims
            context: Background context (RAG results, etc.)

        Returns:
            DebateResult with consensus findings
        """
        if len(self.clients) < 2:
            logger.warning("Debate requires at least 2 participants")
            return DebateResult(
                topic=topic,
                rounds=0,
                debate_summary="Insufficient participants for debate",
            )

        # Initialize claims
        claims = []
        if initial_claims:
            for c in initial_claims:
                claims.append(DebateClaim(
                    text=c.get("text", ""),
                    proposer=c.get("proposer", "initial"),
                    evidence=c.get("evidence", []),
                ))

        # Run debate rounds
        for round_num in range(1, self.config.max_rounds + 1):
            logger.info(f"Starting debate round {round_num}")

            round_result = self._run_round(
                round_number=round_num,
                topic=topic,
                claims=claims,
                context=context,
            )

            self.debate_history.append(round_result)
            claims = round_result.claims

            # Check if consensus reached
            if self._check_consensus(claims):
                logger.info(f"Consensus reached at round {round_num}")
                break

        # Compile final result
        return self._compile_result(topic, claims)

    def _run_round(
        self,
        round_number: int,
        topic: str,
        claims: list[DebateClaim],
        context: str,
    ) -> DebateRound:
        """Run a single debate round.

        Round structure:
        1. Proposal phase: Generate new claims
        2. Critique phase: Challenge existing claims
        3. Support phase: Provide supporting evidence
        4. Voting phase: Vote on each claim
        """
        participants = list(self.clients.keys())

        # Phase 1: Proposal (if first round or revision enabled)
        if round_number == 1 or self.config.enable_revision:
            claims = self._proposal_phase(topic, claims, context)

        # Phase 2: Critique
        claims = self._critique_phase(topic, claims, context)

        # Phase 3: Support
        claims = self._support_phase(topic, claims, context)

        # Phase 4: Voting
        claims = self._voting_phase(claims)

        # Update claim statuses
        for claim in claims:
            claim.status = self._determine_status(claim)

        return DebateRound(
            round_number=round_number,
            topic=topic,
            claims=claims,
            participants=participants,
            summary=f"Round {round_number}: {len(claims)} claims debated",
        )

    def _proposal_phase(
        self,
        topic: str,
        existing_claims: list[DebateClaim],
        context: str,
    ) -> list[DebateClaim]:
        """Generate new claims from each participant."""
        claims = existing_claims.copy()

        proposer_name = self.config.default_proposer
        if proposer_name not in self.clients:
            proposer_name = list(self.clients.keys())[0]

        proposer = self.clients[proposer_name]

        prompt = f"""Topic: {topic}

Context:
{context[:2000] if context else "No additional context."}

Existing claims:
{self._format_claims(claims)}

Based on the topic and context, propose scientific claims.
For each claim:
1. State the claim clearly
2. Provide supporting evidence or reasoning
3. Assign confidence (0-1)

Format as JSON:
{{"claims": [{{"text": "claim text", "evidence": ["evidence 1"], "confidence": 0.8}}]}}"""

        try:
            if hasattr(proposer, "generate_json"):
                result = proposer.generate_json(prompt)
            else:
                response = proposer.generate(prompt)
                result = self._parse_json_response(response)

            for c in result.get("claims", []):
                claims.append(DebateClaim(
                    text=c.get("text", ""),
                    proposer=proposer_name,
                    evidence=[{"type": "reasoning", "content": e} for e in c.get("evidence", [])],
                ))

        except Exception as e:
            logger.warning(f"Proposal phase failed for {proposer_name}: {e}")

        return claims

    def _critique_phase(
        self,
        topic: str,
        claims: list[DebateClaim],
        context: str,
    ) -> list[DebateClaim]:
        """Have critics challenge claims."""
        critic_name = self.config.default_critic
        if critic_name not in self.clients:
            # Use different model than proposer
            critic_name = [k for k in self.clients.keys() if k != self.config.default_proposer]
            critic_name = critic_name[0] if critic_name else list(self.clients.keys())[0]

        critic = self.clients[critic_name]

        for claim in claims:
            if claim.status == ClaimStatus.REJECTED:
                continue

            prompt = f"""Topic: {topic}

Claim to evaluate:
"{claim.text}"

Proposed by: {claim.proposer}
Evidence provided: {claim.evidence}

Context:
{context[:1000] if context else "No additional context."}

As a scientific critic, evaluate this claim:
1. Is the claim scientifically accurate?
2. Is the evidence sufficient?
3. Are there potential errors or oversimplifications?
4. What additional evidence would strengthen/weaken it?

Format as JSON:
{{"critique": "your critique", "issues": ["issue 1"], "strength": 0.7, "recommendation": "accept/revise/reject"}}"""

            try:
                if hasattr(critic, "generate_json"):
                    result = critic.generate_json(prompt)
                else:
                    response = critic.generate(prompt)
                    result = self._parse_json_response(response)

                claim.critiques.append({
                    "critic": critic_name,
                    "critique": result.get("critique", ""),
                    "issues": result.get("issues", []),
                    "strength": result.get("strength", 0.5),
                    "recommendation": result.get("recommendation", "revise"),
                })

            except Exception as e:
                logger.warning(f"Critique failed for claim {claim.id}: {e}")

        return claims

    def _support_phase(
        self,
        topic: str,
        claims: list[DebateClaim],
        context: str,
    ) -> list[DebateClaim]:
        """Have supporters provide additional evidence."""
        # Use third model if available, otherwise skip
        used_models = {self.config.default_proposer, self.config.default_critic}
        supporter_candidates = [k for k in self.clients.keys() if k not in used_models]

        if not supporter_candidates:
            return claims  # No third model for support

        supporter_name = supporter_candidates[0]
        supporter = self.clients[supporter_name]

        for claim in claims:
            if claim.status == ClaimStatus.REJECTED:
                continue

            # Only support claims that received critique
            if not claim.critiques:
                continue

            prompt = f"""Topic: {topic}

Claim: "{claim.text}"

Critiques received:
{claim.critiques}

Can you provide additional evidence or context that either:
1. Supports the claim (addresses the critiques)
2. Confirms the critiques (additional counterevidence)

Be objective and cite sources when possible.

Format as JSON:
{{"position": "support/oppose", "evidence": ["evidence points"], "reasoning": "explanation"}}"""

            try:
                if hasattr(supporter, "generate_json"):
                    result = supporter.generate_json(prompt)
                else:
                    response = supporter.generate(prompt)
                    result = self._parse_json_response(response)

                claim.supports.append({
                    "supporter": supporter_name,
                    "position": result.get("position", "neutral"),
                    "evidence": result.get("evidence", []),
                    "reasoning": result.get("reasoning", ""),
                })

            except Exception as e:
                logger.warning(f"Support phase failed for claim {claim.id}: {e}")

        return claims

    def _voting_phase(self, claims: list[DebateClaim]) -> list[DebateClaim]:
        """All participants vote on each claim."""
        for claim in claims:
            if claim.status == ClaimStatus.REJECTED:
                continue

            for model_name, client in self.clients.items():
                # Skip if this model proposed the claim (conflict of interest)
                if model_name == claim.proposer:
                    continue

                prompt = f"""Vote on this scientific claim:

Claim: "{claim.text}"

Critiques: {claim.critiques}
Supporting evidence: {claim.supports}

Based on the debate, cast your vote:
- SUPPORT: Claim is scientifically sound
- OPPOSE: Claim has significant issues
- NEEDS_EVIDENCE: Claim requires more evidence

Format: {{"vote": "support/oppose/needs_evidence", "reason": "brief reason"}}"""

                try:
                    if hasattr(client, "generate_json"):
                        result = client.generate_json(prompt)
                    else:
                        response = client.generate(prompt)
                        result = self._parse_json_response(response)

                    vote_str = result.get("vote", "abstain").upper()
                    vote = VoteType.SUPPORT if "SUPPORT" in vote_str else \
                           VoteType.OPPOSE if "OPPOSE" in vote_str else \
                           VoteType.NEEDS_EVIDENCE if "EVIDENCE" in vote_str else \
                           VoteType.ABSTAIN

                    claim.votes[model_name] = vote

                except Exception as e:
                    logger.warning(f"Voting failed for {model_name} on claim {claim.id}: {e}")
                    claim.votes[model_name] = VoteType.ABSTAIN

            # Calculate final confidence based on votes
            claim.final_confidence = self._calculate_confidence(claim)

        return claims

    def _calculate_confidence(self, claim: DebateClaim) -> float:
        """Calculate confidence based on votes and critiques."""
        if not claim.votes:
            return 0.5

        support_count = sum(1 for v in claim.votes.values() if v == VoteType.SUPPORT)
        oppose_count = sum(1 for v in claim.votes.values() if v == VoteType.OPPOSE)
        total = len(claim.votes)

        if total == 0:
            return 0.5

        base_confidence = support_count / total

        # Adjust for critique strength
        if claim.critiques:
            avg_strength = sum(c.get("strength", 0.5) for c in claim.critiques) / len(claim.critiques)
            base_confidence = (base_confidence + avg_strength) / 2

        return round(base_confidence, 2)

    def _determine_status(self, claim: DebateClaim) -> ClaimStatus:
        """Determine claim status based on votes."""
        if not claim.votes:
            return ClaimStatus.PROPOSED

        support_ratio = sum(1 for v in claim.votes.values() if v == VoteType.SUPPORT) / len(claim.votes)
        oppose_ratio = sum(1 for v in claim.votes.values() if v == VoteType.OPPOSE) / len(claim.votes)
        evidence_ratio = sum(1 for v in claim.votes.values() if v == VoteType.NEEDS_EVIDENCE) / len(claim.votes)

        if support_ratio >= self.config.consensus_threshold:
            return ClaimStatus.CONSENSUS_REACHED
        elif oppose_ratio >= self.config.consensus_threshold:
            return ClaimStatus.REJECTED
        elif evidence_ratio >= 0.5:
            return ClaimStatus.NEEDS_REVISION
        else:
            return ClaimStatus.UNDER_DEBATE

    def _check_consensus(self, claims: list[DebateClaim]) -> bool:
        """Check if consensus reached on all claims."""
        if not claims:
            return True

        resolved = sum(1 for c in claims if c.status in [
            ClaimStatus.CONSENSUS_REACHED,
            ClaimStatus.REJECTED,
        ])

        return resolved == len(claims)

    def _compile_result(self, topic: str, claims: list[DebateClaim]) -> DebateResult:
        """Compile final debate result."""
        consensus = []
        rejected = []
        unresolved = []

        for claim in claims:
            claim_dict = {
                "text": claim.text,
                "proposer": claim.proposer,
                "confidence": claim.final_confidence,
                "votes": {k: v.value for k, v in claim.votes.items()},
                "critiques_count": len(claim.critiques),
            }

            if claim.status == ClaimStatus.CONSENSUS_REACHED:
                consensus.append(claim_dict)
            elif claim.status == ClaimStatus.REJECTED:
                rejected.append(claim_dict)
            else:
                unresolved.append(claim_dict)

        overall_confidence = (
            sum(c["confidence"] for c in consensus) / len(consensus)
            if consensus else 0.0
        )

        return DebateResult(
            topic=topic,
            rounds=len(self.debate_history),
            consensus_claims=consensus,
            rejected_claims=rejected,
            unresolved_claims=unresolved,
            overall_confidence=overall_confidence,
            debate_summary=self._generate_summary(topic, claims),
        )

    def _generate_summary(self, topic: str, claims: list[DebateClaim]) -> str:
        """Generate human-readable debate summary."""
        consensus_count = sum(1 for c in claims if c.status == ClaimStatus.CONSENSUS_REACHED)
        rejected_count = sum(1 for c in claims if c.status == ClaimStatus.REJECTED)
        unresolved_count = len(claims) - consensus_count - rejected_count

        return (
            f"Debate on '{topic}' completed.\n"
            f"Participants: {', '.join(self.clients.keys())}\n"
            f"Rounds: {len(self.debate_history)}\n"
            f"Results: {consensus_count} consensus, {rejected_count} rejected, {unresolved_count} unresolved"
        )

    def _format_claims(self, claims: list[DebateClaim]) -> str:
        """Format claims for prompt."""
        if not claims:
            return "No existing claims."

        formatted = []
        for i, c in enumerate(claims, 1):
            formatted.append(f"{i}. [{c.proposer}] {c.text}")

        return "\n".join(formatted)

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from response text."""
        import json

        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {}


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_debate(
    topic: str,
    gemini_client: Any = None,
    grok_client: Any = None,
    deepseek_client: Any = None,
    claude_client: Any = None,
    openai_client: Any = None,
    qwen_client: Any = None,
) -> DebateResult:
    """Run a quick debate with available clients.

    Default 3-LLM configuration:
    - Gemini: Proposes claims (fast, free)
    - Grok: Critiques claims (reasoning, xAI)
    - DeepSeek: Provides third vote (independent, cheap)

    Args:
        topic: Topic to debate
        gemini_client: Gemini client instance (proposer)
        grok_client: Grok client instance (critic)
        deepseek_client: DeepSeek client instance (judge/third vote)
        claude_client: Claude client instance (fallback)
        openai_client: OpenAI GPT-4 client instance (fallback)
        qwen_client: Qwen-VL client instance (fallback for multimodal)

    Returns:
        DebateResult
    """
    clients = {}

    # Proposer: Gemini
    if gemini_client:
        clients["gemini"] = gemini_client

    # Critic: Grok (fallback to Claude)
    if grok_client:
        clients["grok"] = grok_client
    elif claude_client:
        clients["claude"] = claude_client

    # Judge: DeepSeek (fallback to OpenAI, then Qwen)
    if deepseek_client:
        clients["deepseek"] = deepseek_client
    elif openai_client:
        clients["openai"] = openai_client
    elif qwen_client:
        clients["qwen"] = qwen_client

    # Configure based on available clients
    critic = "grok" if "grok" in clients else "claude" if "claude" in clients else "gemini"
    judge = "deepseek" if "deepseek" in clients else "openai" if "openai" in clients else "qwen" if "qwen" in clients else "gemini"

    config = DebateConfig(
        default_proposer="gemini",
        default_critic=critic,
        default_judge=judge,
    )

    engine = DebateEngine(clients=clients, config=config)
    return engine.debate(topic=topic)
