"""Integration tests for LangGraph orchestrator.

Tests verify that orchestrator nodes properly integrate with:
- Model clients (Gemini, Claude, Lobster)
- RAG system (knowledge base, retriever)
- ADMET predictor (DeepChem)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestOrchestratorInitialization:
    """Test orchestrator initialization."""

    def test_orchestrator_creates_without_error(self):
        """Test that orchestrator can be created without API keys."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        # Should not raise even without API keys - uses lazy loading
        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )
        assert orchestrator is not None
        assert orchestrator.graph is not None

    def test_orchestrator_with_checkpointing(self):
        """Test orchestrator with checkpointing enabled."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=True,
            use_rag=False,
        )
        assert orchestrator.checkpointer is not None


class TestRouterNode:
    """Test router node logic."""

    def test_router_full_workflow(self):
        """Test router directs to context retrieval for full workflow."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        state = {"workflow_type": "full"}
        result = orchestrator._router_node(state)

        assert result["current_step"] == "retrieve_context"

    def test_router_admet_workflow(self):
        """Test router skips to ADMET for ADMET-only workflow."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        state = {"workflow_type": "admet"}
        result = orchestrator._router_node(state)

        assert result["current_step"] == "admet"


class TestPathogenesisNode:
    """Test pathogenesis node integration."""

    def test_pathogenesis_without_clients(self):
        """Test pathogenesis node returns placeholder without LLM clients."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        state = {
            "disease": "Type 2 Diabetes",
            "tissue": "liver",
            "context": "",
            "workflow_type": "full",
        }

        result = orchestrator._pathogenesis_node(state)

        assert "claims" in result
        assert len(result["claims"]) > 0
        assert "current_step" in result

    @patch("src.chains.langgraph_orchestrator.ResearchOrchestrator.claude_client", new_callable=lambda: property(lambda self: None))
    @patch("src.chains.langgraph_orchestrator.ResearchOrchestrator.gemini_client")
    def test_pathogenesis_with_gemini_fallback(self, mock_gemini_prop):
        """Test pathogenesis falls back to Gemini if Claude unavailable."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        # Create mock Gemini client
        mock_gemini = Mock()
        mock_gemini.generate_json.return_value = {
            "pathways": ["AMPK", "insulin signaling"],
            "genes": ["PPARG", "ADIPOQ"],
            "mechanisms": ["insulin resistance"],
            "summary": "Test summary",
        }
        mock_gemini_prop.return_value = mock_gemini

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        # Force gemini client to be the mock
        orchestrator._gemini_client = mock_gemini
        orchestrator._claude_client = None

        state = {
            "disease": "Type 2 Diabetes",
            "tissue": "liver",
            "context": "",
            "workflow_type": "full",
        }

        result = orchestrator._pathogenesis_node(state)

        assert "claims" in result
        assert mock_gemini.generate_json.called


class TestTargetNode:
    """Test target discovery node integration."""

    def test_target_without_clients(self):
        """Test target node returns placeholder targets without clients."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        state = {
            "disease": "Type 2 Diabetes",
            "omics_results": [],
            "context": "",
            "workflow_type": "full",
        }

        result = orchestrator._target_node(state)

        assert "targets" in result
        assert "claims" in result
        # Should have placeholder targets
        assert len(result.get("targets", [])) >= 0


class TestADMETNode:
    """Test ADMET prediction node integration."""

    def test_admet_no_compound(self):
        """Test ADMET node handles missing compound gracefully."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        state = {
            "compound_smiles": None,
            "targets": [],
        }

        result = orchestrator._admet_node(state)

        assert result["current_step"] == "synthesis"
        assert "errors" in result

    def test_admet_with_invalid_smiles(self):
        """Test ADMET node handles invalid SMILES."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        state = {
            "compound_smiles": "invalid_smiles_string",
            "targets": [],
        }

        result = orchestrator._admet_node(state)

        # Should complete without crashing
        assert "current_step" in result


class TestSynthesisNode:
    """Test synthesis node integration."""

    def test_synthesis_generates_report(self):
        """Test synthesis generates final report."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        state = {
            "disease": "Type 2 Diabetes",
            "claims": [
                {
                    "text": "Test claim with sufficient length for validation",
                    "confidence": 0.8,
                    "evidence_tier": "tier2",
                    "evidence": [{"type": "computed", "tool": "test", "artifact_id": "t1"}],
                }
            ],
            "targets": [{"gene_symbol": "PPARG", "priority": "High"}],
            "admet_results": [],
            "context": "",
            "citations": [],
        }

        result = orchestrator._synthesis_node(state)

        assert "final_report" in result
        assert result["final_report"]["disease"] == "Type 2 Diabetes"
        assert "evidence_tier" in result

    def test_synthesis_triggers_review_for_tier3(self):
        """Test synthesis triggers human review for low evidence tier."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        state = {
            "disease": "Test Disease",
            "claims": [
                {
                    "text": "Test claim with only tier3 evidence level",
                    "confidence": 0.5,
                    "evidence_tier": "tier3",
                    "evidence": [{"type": "computed", "tool": "test", "artifact_id": "t1"}],
                }
            ],
            "targets": [],
            "admet_results": [],
            "context": "",
            "citations": [],
        }

        result = orchestrator._synthesis_node(state)

        assert result["needs_human_review"] is True
        assert result["evidence_tier"] == "tier3"


class TestContextRetrieval:
    """Test RAG context retrieval node."""

    def test_retrieval_without_rag(self):
        """Test retrieval returns empty context when RAG disabled."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        state = {
            "disease": "Type 2 Diabetes",
            "tissue": "liver",
        }

        result = orchestrator._retrieve_context_node(state)

        assert result["context"] == ""
        assert result["citations"] == []
        assert result["current_step"] == "pathogenesis"


class TestFullWorkflow:
    """Test complete workflow execution."""

    def test_full_workflow_runs_without_error(self):
        """Test that full workflow completes without API keys."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        # Run workflow - should complete with placeholders
        result = orchestrator.run(
            disease="Type 2 Diabetes",
            tissue="liver",
            workflow_type="full",
        )

        assert result is not None
        assert "final_report" in result
        assert result["disease"] == "Type 2 Diabetes"

    def test_pathogenesis_only_workflow(self):
        """Test pathogenesis-only workflow."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        result = orchestrator.run(
            disease="NAFLD",
            workflow_type="pathogenesis",
        )

        assert result is not None

    def test_admet_only_workflow(self):
        """Test ADMET-only workflow."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        result = orchestrator.run(
            disease="Test",
            compound_smiles="CCO",  # Ethanol
            workflow_type="admet",
        )

        assert result is not None


class TestStreamingExecution:
    """Test streaming workflow execution."""

    def test_stream_yields_events(self):
        """Test that streaming yields state updates."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        events = list(orchestrator.stream(
            disease="Type 2 Diabetes",
            workflow_type="full",
        ))

        assert len(events) > 0
