#!/usr/bin/env python3
"""AgingResearchAI Skill Runner for Claude Code integration.

Quick commands for common research tasks.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")


def cmd_search(args):
    """Search PubMed literature."""
    from src.rag import KnowledgeBase, PubMedLoader, HybridRetriever

    print(f"Searching PubMed for: {args.query}")

    kb = KnowledgeBase(persist_directory="data/embeddings")

    # Load if needed
    if args.load:
        loader = PubMedLoader(query=args.query, max_results=args.max_results)
        docs = loader.load()
        kb.add_documents(docs, collection="literature")
        print(f"Loaded {len(docs)} documents")

    # Search
    retriever = HybridRetriever(kb, collection="literature")
    results = retriever.retrieve(args.query, top_k=args.top_k)

    print(f"\n{'='*60}")
    print(f"Found {len(results)} results:")
    print(f"{'='*60}\n")

    for i, r in enumerate(results, 1):
        meta = r.document.metadata
        pmid = getattr(meta, 'pmid', 'N/A')
        title = getattr(meta, 'title', 'Untitled')
        print(f"{i}. [PMID:{pmid}] {title}")
        print(f"   Score: {r.score:.3f}")
        print(f"   {r.document.content[:200]}...")
        print()


def cmd_debate(args):
    """Run multi-LLM debate on a claim."""
    from src.chains import DebateEngine, DebateConfig
    from src.models import GeminiClient, DeepSeekClient

    print(f"Starting debate on: {args.topic}")
    print(f"Claim: {args.claim}")
    print()

    # Initialize clients
    clients = {}
    try:
        clients["gemini"] = GeminiClient()
        print("[OK] Gemini (proposer)")
    except Exception as e:
        print(f"[X] Gemini: {e}")

    try:
        from src.models import GrokClient
        clients["grok"] = GrokClient()
        print("[OK] Grok (critic)")
    except Exception as e:
        print(f"[X] Grok: {e}")

    try:
        clients["deepseek"] = DeepSeekClient()
        print("[OK] DeepSeek (judge)")
    except Exception as e:
        print(f"[X] DeepSeek: {e}")

    if len(clients) < 2:
        print("\nError: Need at least 2 LLM clients for debate")
        return

    engine = DebateEngine(
        clients=clients,
        config=DebateConfig(max_rounds=args.rounds),
    )

    initial_claims = []
    if args.claim:
        initial_claims.append({
            "text": args.claim,
            "proposer": "user",
        })

    print(f"\n{'='*60}")
    print("Running debate...")
    print(f"{'='*60}\n")

    result = engine.debate(
        topic=args.topic,
        initial_claims=initial_claims,
    )

    print(f"\nDebate completed in {result.rounds} rounds")
    print(f"\nConsensus claims ({len(result.consensus_claims)}):")
    for c in result.consensus_claims:
        print(f"  [OK] {c['text']} ({c['confidence']:.0%})")

    print(f"\nRejected claims ({len(result.rejected_claims)}):")
    for c in result.rejected_claims:
        print(f"  [X] {c['text']}")

    print(f"\nUnresolved ({len(result.unresolved_claims)}):")
    for c in result.unresolved_claims:
        print(f"  [?] {c['text']}")

    print(f"\nOverall confidence: {result.overall_confidence:.0%}")


def cmd_admet(args):
    """Predict ADMET properties."""
    from src.admet import quick_admet_check, DeepChemPredictor

    print(f"ADMET analysis for: {args.smiles}")
    print()

    if args.quick:
        result = quick_admet_check(args.smiles)
        print("Quick ADMET Check (RDKit):")
        print(f"  Molecular Weight: {result['molecular_weight']:.2f}")
        print(f"  LogP: {result['logp']:.2f}")
        print(f"  H-bond donors: {result['hbd']}")
        print(f"  H-bond acceptors: {result['hba']}")
        print(f"  TPSA: {result['tpsa']:.2f}")
        print(f"  Rotatable bonds: {result['rotatable_bonds']}")
        print()
        print(f"  Lipinski: {'PASS' if result['lipinski_passes'] else 'FAIL'} ({result['lipinski_violations']} violations)")
        print(f"  Est. Absorption: {result['estimated_absorption']}")
        print(f"  Est. BBB: {result['estimated_bbb']}")
        print(f"  Overall Risk: {result['overall_risk']}")
    else:
        try:
            predictor = DeepChemPredictor()
            result = predictor.predict(args.smiles)

            print("Full ADMET Prediction:")
            print(f"\nPhysicochemical:")
            print(f"  MW: {result.physicochemical.molecular_weight:.2f}")
            print(f"  LogP: {result.physicochemical.logp:.2f}")

            print(f"\nLipinski: {'PASS' if result.lipinski.passes else 'FAIL'}")

            if result.tox21:
                print(f"\nTox21 High-Risk: {', '.join(result.tox21.high_risk_endpoints) or 'None'}")

            print(f"\nOverall Risk: {result.overall_risk}")
            print(f"Flags: {', '.join(result.flags) or 'None'}")
            print(f"\nRecommendations:")
            for rec in result.recommendations:
                print(f"  - {rec}")

        except ImportError:
            print("DeepChem not installed. Using quick check instead.")
            cmd_admet(argparse.Namespace(smiles=args.smiles, quick=True))


def cmd_pipeline(args):
    """Run research pipeline."""
    from src.chains import ResearchOrchestrator

    print(f"Running {args.workflow} pipeline for {args.disease} in {args.tissue}")
    print()

    orchestrator = ResearchOrchestrator(
        enable_debate=args.debate,
        use_rag=args.rag,
    )

    result = orchestrator.run(
        disease=args.disease,
        tissue=args.tissue,
        workflow_type=args.workflow,
    )

    print(f"\n{'='*60}")
    print(f"Pipeline Complete")
    print(f"{'='*60}\n")

    print(f"Summary: {result.get('summary', 'N/A')}")
    print(f"\nClaims ({len(result.get('claims', []))}):")
    for c in result.get('claims', []):
        print(f"  [{c.get('status', '?')}] {c.get('text', '')}")


def cmd_server(args):
    """Start API server."""
    import uvicorn

    print(f"Starting AgingResearchAI API server on {args.host}:{args.port}")
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_costs(args):
    """Show cost tracking."""
    from src.monitoring import get_cost_tracker, LLM_PRICING

    tracker = get_cost_tracker()

    print("LLM Pricing (per 1M tokens):")
    print(f"{'='*50}")
    for model, pricing in LLM_PRICING.items():
        if not model.startswith("gpt") and not model.startswith("claude-"):
            print(f"  {model:15} In: ${pricing['input']:.2f}  Out: ${pricing['output']:.2f}")

    print(f"\n{'='*50}")
    budget = tracker.check_budget()
    print(f"Daily:   ${budget['daily']['spent_usd']:.2f} / ${budget['daily']['budget_usd']:.2f} ({budget['daily']['percentage']:.0f}%)")
    print(f"Monthly: ${budget['monthly']['spent_usd']:.2f} / ${budget['monthly']['budget_usd']:.2f} ({budget['monthly']['percentage']:.0f}%)")

    if budget['alerts']:
        print("\nAlerts:")
        for alert in budget['alerts']:
            print(f"  [{alert['level']}] {alert['message']}")

    tips = tracker.get_cost_optimization_tips()
    print("\nOptimization Tips:")
    for tip in tips:
        print(f"  - {tip}")


def main():
    parser = argparse.ArgumentParser(
        description="AgingResearchAI Skill Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search PubMed literature")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--load", action="store_true", help="Load new papers first")
    search_parser.add_argument("--max-results", type=int, default=50)
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.set_defaults(func=cmd_search)

    # Debate command
    debate_parser = subparsers.add_parser("debate", help="Run multi-LLM debate")
    debate_parser.add_argument("topic", help="Debate topic")
    debate_parser.add_argument("--claim", help="Initial claim to evaluate")
    debate_parser.add_argument("--rounds", type=int, default=2)
    debate_parser.set_defaults(func=cmd_debate)

    # ADMET command
    admet_parser = subparsers.add_parser("admet", help="ADMET prediction")
    admet_parser.add_argument("smiles", help="SMILES string")
    admet_parser.add_argument("--quick", action="store_true", help="Quick RDKit-only check")
    admet_parser.set_defaults(func=cmd_admet)

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run research pipeline")
    pipeline_parser.add_argument("disease", help="Disease (T2D, NAFLD, Sarcopenia)")
    pipeline_parser.add_argument("--tissue", default="liver")
    pipeline_parser.add_argument("--workflow", default="pathogenesis",
                                  choices=["pathogenesis", "target_discovery", "full"])
    pipeline_parser.add_argument("--no-debate", dest="debate", action="store_false")
    pipeline_parser.add_argument("--no-rag", dest="rag", action="store_false")
    pipeline_parser.set_defaults(func=cmd_pipeline)

    # Server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--host", default="0.0.0.0")
    server_parser.add_argument("--port", type=int, default=8000)
    server_parser.add_argument("--reload", action="store_true")
    server_parser.set_defaults(func=cmd_server)

    # Costs command
    costs_parser = subparsers.add_parser("costs", help="Show cost tracking")
    costs_parser.set_defaults(func=cmd_costs)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
