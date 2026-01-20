#!/usr/bin/env python
"""Main entry point for AgingResearchAI pipeline.

Usage:
    python scripts/run_pipeline.py --disease "Type 2 Diabetes" --tissue "liver"
    python scripts/run_pipeline.py --disease "NAFLD" --quick-check
    python scripts/run_pipeline.py --compound "CCO" --admet
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def run_pathogenesis(disease: str, tissue: str | None = None, quick: bool = False):
    """Run pathogenesis discovery pipeline."""
    from src.agents.pathogenesis_agent import PathogenesisAgent, QuickPathogenesisCheck

    if quick:
        print(f"Running quick pathogenesis check for {disease}...")
        agent = QuickPathogenesisCheck()
        result = agent.quick_check(disease)
    else:
        print(f"Running full pathogenesis discovery for {disease}...")
        agent = PathogenesisAgent()
        result = agent.discover(disease, tissue=tissue)

    # Output result
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(json.dumps(result.model_dump(), indent=2, default=str))

    return result


def run_target_discovery(disease: str, genes: list[str] | None = None):
    """Run target discovery pipeline."""
    from src.models.claude_client import ClaudeClient

    print(f"Running target discovery for {disease}...")

    client = ClaudeClient()

    # Example targets (in real use, these come from pathogenesis analysis)
    targets = [
        {
            "gene_symbol": gene,
            "druggability_score": 0.7,
            "safety_score": 0.8,
            "evidence_score": 0.6,
        }
        for gene in (genes or ["PPARG", "AMPK", "SIRT1"])
    ]

    criteria = {
        "min_druggability": 0.5,
        "min_safety": 0.6,
        "disease_relevance": disease,
    }

    result = client.prioritize_targets(targets, criteria)

    print("\n" + "=" * 60)
    print("TARGET PRIORITIZATION RESULT")
    print("=" * 60)
    print(json.dumps(result.model_dump(), indent=2, default=str))

    return result


def run_admet(smiles: str, compound_id: str = "test_compound"):
    """Run ADMET analysis pipeline."""
    from src.models.claude_client import ClaudeClient

    print(f"Running ADMET analysis for {compound_id}...")

    client = ClaudeClient()

    # Example ADMET results (in real use, computed by RDKit/pkCSM)
    admet_results = {
        "physicochemical": {
            "mw": 180.16,
            "logp": -0.07,
            "hbd": 1,
            "hba": 2,
            "tpsa": 20.23,
        },
        "absorption": {
            "intestinal_absorption": "High",
            "caco2_permeability": 1.2,
        },
        "toxicity": {
            "ames": "Negative",
            "herg": "Low risk",
        },
    }

    result = client.interpret_admet(
        compound_id=compound_id,
        smiles=smiles,
        admet_results=admet_results,
    )

    print("\n" + "=" * 60)
    print("ADMET RESULT")
    print("=" * 60)
    print(json.dumps(result.model_dump(), indent=2, default=str))

    return result


def run_experiment_design(hypothesis: str, target: str):
    """Design validation experiments."""
    from src.models.claude_client import ClaudeClient

    print(f"Designing experiments for: {hypothesis}...")

    client = ClaudeClient()

    constraints = {
        "budget_tier": "$$",
        "timeline": "6 months",
        "available_models": ["cell lines", "primary cells", "mouse models"],
    }

    result = client.design_experiments(
        hypothesis=hypothesis,
        target=target,
        constraints=constraints,
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT DESIGN RESULT")
    print("=" * 60)
    print(json.dumps(result.model_dump(), indent=2, default=str))

    return result


def check_lobster_status():
    """Check Lobster AI status."""
    from src.models.lobster_client import LobsterClient

    print("Checking Lobster AI status...")
    client = LobsterClient()
    status = client.check_status()

    print("\n" + "=" * 60)
    print("LOBSTER STATUS")
    print("=" * 60)
    print(json.dumps(status, indent=2))

    if not status.get("valid"):
        print("\nLobster is not configured. Run: lobster init")

    return status


def main():
    parser = argparse.ArgumentParser(
        description="AgingResearchAI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pathogenesis discovery
    python scripts/run_pipeline.py --disease "Type 2 Diabetes" --tissue "liver"

    # Quick check (Gemini only, cost-effective)
    python scripts/run_pipeline.py --disease "NAFLD" --quick-check

    # Target prioritization
    python scripts/run_pipeline.py --disease "T2D" --targets PPARG,AMPK,SIRT1

    # ADMET analysis
    python scripts/run_pipeline.py --compound "CCO" --admet

    # Experiment design
    python scripts/run_pipeline.py --hypothesis "SIRT1 activation reduces hepatic steatosis" --target "SIRT1"

    # Check Lobster status
    python scripts/run_pipeline.py --check-lobster
        """,
    )

    parser.add_argument("--disease", type=str, help="Disease to analyze")
    parser.add_argument("--tissue", type=str, help="Target tissue")
    parser.add_argument("--quick-check", action="store_true", help="Run quick check only (Gemini)")
    parser.add_argument("--targets", type=str, help="Comma-separated list of gene targets")
    parser.add_argument("--compound", type=str, help="SMILES string for ADMET analysis")
    parser.add_argument("--admet", action="store_true", help="Run ADMET analysis")
    parser.add_argument("--hypothesis", type=str, help="Hypothesis for experiment design")
    parser.add_argument("--target", type=str, help="Target gene for experiment design")
    parser.add_argument("--check-lobster", action="store_true", help="Check Lobster AI status")

    args = parser.parse_args()

    # Check Lobster status
    if args.check_lobster:
        check_lobster_status()
        return

    # ADMET analysis
    if args.admet and args.compound:
        run_admet(args.compound)
        return

    # Experiment design
    if args.hypothesis and args.target:
        run_experiment_design(args.hypothesis, args.target)
        return

    # Target discovery
    if args.disease and args.targets:
        genes = args.targets.split(",")
        run_target_discovery(args.disease, genes)
        return

    # Pathogenesis discovery
    if args.disease:
        run_pathogenesis(args.disease, args.tissue, quick=args.quick_check)
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
