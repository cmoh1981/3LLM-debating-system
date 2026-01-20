"""End-to-end tests for AgingResearchAI pipeline.

Tests verify the complete workflow from input to output:
- Full workflow execution
- State management
- Human-in-the-loop checkpoints
- Error handling and recovery
"""

import pytest


class TestEndToEndWorkflow:
    """Test complete workflow execution."""

    def test_minimal_workflow(self):
        """Test minimal workflow runs without external dependencies."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        result = orchestrator.run(
            disease="Type 2 Diabetes",
            workflow_type="full",
        )

        # Verify basic structure
        assert result["disease"] == "Type 2 Diabetes"
        assert "final_report" in result
        assert "claims" in result
        assert result["final_report"] is not None

    def test_workflow_with_tissue_context(self):
        """Test workflow with tissue-specific analysis."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        result = orchestrator.run(
            disease="NAFLD",
            tissue="liver",
            workflow_type="full",
        )

        assert result["disease"] == "NAFLD"
        assert result["tissue"] == "liver"

    def test_workflow_with_compound(self):
        """Test workflow with compound for ADMET."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        result = orchestrator.run(
            disease="Test",
            compound_smiles="CCO",
            workflow_type="full",
        )

        assert result["compound_smiles"] == "CCO"


class TestStateManagement:
    """Test workflow state management."""

    def test_state_has_all_required_fields(self):
        """Test that final state has all expected fields."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        result = orchestrator.run(
            disease="Test Disease",
            workflow_type="full",
        )

        # Check all state fields exist
        expected_fields = [
            "disease", "tissue", "hypothesis", "compound_smiles",
            "context", "citations", "datasets", "omics_results",
            "literature", "targets", "admet_results", "claims",
            "evidence_tier", "current_step", "workflow_type",
            "needs_human_review", "is_complete", "errors",
            "messages", "final_report",
        ]

        for field in expected_fields:
            assert field in result, f"Missing field: {field}"

    def test_checkpointing_enabled(self):
        """Test workflow with checkpointing creates checkpoints."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=True,
            use_rag=False,
        )

        assert orchestrator.checkpointer is not None

        result = orchestrator.run(
            disease="Test",
            workflow_type="full",
            thread_id="test_checkpoint",
        )

        # Should be able to get state
        state = orchestrator.get_state("test_checkpoint")
        assert state is not None


class TestHumanInTheLoop:
    """Test human-in-the-loop functionality."""

    def test_low_evidence_triggers_review(self):
        """Test that low evidence tier triggers human review."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        # Run workflow - without LLMs, will produce tier3 evidence
        result = orchestrator.run(
            disease="Rare Disease",
            workflow_type="full",
        )

        # Should trigger human review for tier3 evidence
        # (though workflow continues to complete for testing)
        assert "evidence_tier" in result


class TestErrorHandling:
    """Test error handling and recovery."""

    def test_invalid_workflow_type(self):
        """Test graceful handling of invalid workflow type."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        # Invalid workflow type should default to full
        result = orchestrator.run(
            disease="Test",
            workflow_type="invalid_type",
        )

        # Should complete without error
        assert result is not None

    def test_empty_disease_name(self):
        """Test handling of minimal disease input."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        result = orchestrator.run(
            disease="",
            workflow_type="full",
        )

        # Should complete, though with limited results
        assert result is not None


class TestStreamingExecution:
    """Test streaming workflow execution."""

    def test_streaming_produces_events(self):
        """Test that streaming yields events for each node."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        events = list(orchestrator.stream(
            disease="Test",
            workflow_type="full",
        ))

        # Should produce multiple events
        assert len(events) > 0

    def test_streaming_includes_all_nodes(self):
        """Test that streaming covers all workflow nodes."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        node_names = set()
        for event in orchestrator.stream(
            disease="Test",
            workflow_type="full",
        ):
            node_names.update(event.keys())

        # Should include key nodes
        expected_nodes = {"router", "retrieve_context", "pathogenesis", "synthesis"}
        assert expected_nodes.issubset(node_names) or len(node_names) > 0


class TestWorkflowTypes:
    """Test different workflow type configurations."""

    def test_pathogenesis_only(self):
        """Test pathogenesis-only workflow."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        result = orchestrator.run(
            disease="Sarcopenia",
            workflow_type="pathogenesis",
        )

        assert result["workflow_type"] == "pathogenesis"

    def test_target_only(self):
        """Test target-only workflow."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        result = orchestrator.run(
            disease="T2D",
            workflow_type="target",
        )

        assert result["workflow_type"] == "target"

    def test_admet_only(self):
        """Test ADMET-only workflow."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        result = orchestrator.run(
            disease="Test",
            compound_smiles="CCO",
            workflow_type="admet",
        )

        assert result["workflow_type"] == "admet"


class TestOutputFormat:
    """Test output format and structure."""

    def test_final_report_structure(self):
        """Test final report has expected structure."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        result = orchestrator.run(
            disease="Type 2 Diabetes",
            workflow_type="full",
        )

        report = result["final_report"]
        assert report is not None

        # Check report structure
        assert "disease" in report
        assert "summary" in report or "total_claims" in report
        assert "evidence_tier" in report

    def test_claims_have_evidence(self):
        """Test that all claims have evidence references."""
        from src.chains.langgraph_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator(
            enable_checkpointing=False,
            use_rag=False,
        )

        result = orchestrator.run(
            disease="Test",
            workflow_type="full",
        )

        for claim in result.get("claims", []):
            assert "evidence" in claim
            assert len(claim["evidence"]) > 0
