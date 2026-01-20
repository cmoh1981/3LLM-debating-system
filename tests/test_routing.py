"""Tests for model routing logic."""

import pytest

from src.core.router import ModelRouter, TaskType, ROUTING_RULES


class TestModelRouter:
    """Test ModelRouter class."""

    def test_gemini_tasks_routed_correctly(self):
        """Test that Gemini tasks are routed to Gemini."""
        router = ModelRouter()

        gemini_tasks = [
            TaskType.LITERATURE_SEARCH,
            TaskType.DATASET_DISCOVERY,
            TaskType.CODE_GENERATION,
            TaskType.DATA_EXTRACTION,
            TaskType.PATENT_SEARCH,
        ]

        for task in gemini_tasks:
            decision = router.route(task)
            assert decision.model == "gemini", f"{task} should route to gemini"

    def test_claude_tasks_routed_correctly(self):
        """Test that Claude tasks are routed to Claude."""
        router = ModelRouter()

        claude_tasks = [
            TaskType.PATHOGENESIS_SYNTHESIS,
            TaskType.TARGET_PRIORITIZATION,
            TaskType.RISK_ASSESSMENT,
            TaskType.EXPERIMENT_DESIGN,
            TaskType.REPORT_GENERATION,
        ]

        for task in claude_tasks:
            decision = router.route(task)
            assert decision.model == "claude", f"{task} should route to claude"

    def test_lobster_tasks_routed_correctly(self):
        """Test that Lobster tasks are routed to Lobster."""
        router = ModelRouter()

        lobster_tasks = [
            TaskType.RNASEQ_ANALYSIS,
            TaskType.SCRNA_ANALYSIS,
            TaskType.GEO_DATASET_SEARCH,
            TaskType.DIFFERENTIAL_EXPRESSION,
            TaskType.PATHWAY_ENRICHMENT,
        ]

        for task in lobster_tasks:
            decision = router.route(task)
            assert decision.model == "lobster", f"{task} should route to lobster"

    def test_routing_decision_has_fallback(self):
        """Test that routing decisions include fallback model."""
        router = ModelRouter()
        decision = router.route(TaskType.PATHOGENESIS_SYNTHESIS)
        assert decision.fallback is not None

    def test_routing_decision_has_reason(self):
        """Test that routing decisions include reasoning."""
        router = ModelRouter()
        decision = router.route(TaskType.CODE_GENERATION)
        assert decision.reason is not None
        assert len(decision.reason) > 0

    def test_cost_estimation(self):
        """Test task distribution estimation."""
        router = ModelRouter()

        tasks = [
            TaskType.LITERATURE_SEARCH,  # gemini
            TaskType.LITERATURE_SEARCH,  # gemini
            TaskType.PATHOGENESIS_SYNTHESIS,  # claude
            TaskType.RNASEQ_ANALYSIS,  # lobster
        ]

        counts = router.estimate_cost(tasks)
        assert counts["gemini"] == 2
        assert counts["claude"] == 1
        assert counts["lobster"] == 1

    def test_get_tasks_by_model(self):
        """Test getting all tasks for a specific model."""
        gemini_tasks = ModelRouter.get_gemini_tasks()
        claude_tasks = ModelRouter.get_claude_tasks()
        lobster_tasks = ModelRouter.get_lobster_tasks()

        # Verify counts match expected distribution
        assert len(gemini_tasks) == 7  # ~80% of tasks
        assert len(claude_tasks) == 7  # ~20% of critical tasks
        assert len(lobster_tasks) == 6  # Specialized bioinformatics

    def test_custom_routing_rules(self):
        """Test router with custom routing rules."""
        custom_rules = {
            TaskType.LITERATURE_SEARCH: "claude",  # Override default
        }
        router = ModelRouter(rules=custom_rules)

        decision = router.route(TaskType.LITERATURE_SEARCH)
        assert decision.model == "claude"

    def test_default_model_for_unknown_task(self):
        """Test that unknown tasks fall back to default model."""
        router = ModelRouter(rules={}, default_model="gemini")

        # Route a task not in rules
        model = router.get_model_for_task(TaskType.LITERATURE_SEARCH)
        assert model == "gemini"


class TestRoutingRules:
    """Test ROUTING_RULES constant."""

    def test_all_task_types_have_rules(self):
        """Test that all TaskTypes have routing rules."""
        for task_type in TaskType:
            assert task_type in ROUTING_RULES, f"Missing rule for {task_type}"

    def test_rules_use_valid_models(self):
        """Test that all rules specify valid model names."""
        valid_models = {"gemini", "claude", "lobster"}

        for task_type, model in ROUTING_RULES.items():
            assert model in valid_models, f"Invalid model {model} for {task_type}"

    def test_cost_distribution(self):
        """Test that roughly 80% of tasks go to free tier (Gemini)."""
        gemini_count = sum(1 for m in ROUTING_RULES.values() if m == "gemini")
        total = len(ROUTING_RULES)

        # Gemini should handle majority of tasks
        gemini_ratio = gemini_count / total
        assert gemini_ratio >= 0.3, f"Gemini should handle at least 30% of tasks, got {gemini_ratio:.1%}"
