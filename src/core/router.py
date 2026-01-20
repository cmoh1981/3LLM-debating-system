"""Model routing logic for AgingResearchAI.

Routes tasks to the appropriate model based on task type:
- Gemini (free, 80% of tasks): Literature search, code gen, data extraction
- Claude (paid, 20% of tasks): Complex reasoning, synthesis, final reports
- Lobster (specialized): Bioinformatics, RNA-seq, pathway analysis
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class TaskType(str, Enum):
    """Task types supported by the system."""

    # Gemini tasks (free tier, bulk operations)
    LITERATURE_SEARCH = "literature_search"
    DATASET_DISCOVERY = "dataset_discovery"
    CODE_GENERATION = "code_generation"
    DATA_EXTRACTION = "data_extraction"
    PATENT_SEARCH = "patent_search"
    INITIAL_SCREENING = "initial_screening"
    API_ORCHESTRATION = "api_orchestration"

    # Claude tasks (paid, complex reasoning)
    PATHOGENESIS_SYNTHESIS = "pathogenesis_synthesis"
    TARGET_PRIORITIZATION = "target_prioritization"
    RISK_ASSESSMENT = "risk_assessment"
    EXPERIMENT_DESIGN = "experiment_design"
    FINAL_INTERPRETATION = "final_interpretation"
    REPORT_GENERATION = "report_generation"
    EVIDENCE_ADJUDICATION = "evidence_adjudication"

    # Lobster tasks (specialized bioinformatics)
    RNASEQ_ANALYSIS = "rnaseq_analysis"
    SCRNA_ANALYSIS = "scrna_analysis"
    GEO_DATASET_SEARCH = "geo_dataset_search"
    DIFFERENTIAL_EXPRESSION = "differential_expression"
    PATHWAY_ENRICHMENT = "pathway_enrichment"
    QUALITY_CONTROL = "quality_control"


ModelName = Literal["gemini", "claude", "lobster"]


ROUTING_RULES: dict[TaskType, ModelName] = {
    # Gemini (Free, 80% of tasks)
    TaskType.LITERATURE_SEARCH: "gemini",
    TaskType.DATASET_DISCOVERY: "gemini",
    TaskType.CODE_GENERATION: "gemini",
    TaskType.DATA_EXTRACTION: "gemini",
    TaskType.PATENT_SEARCH: "gemini",
    TaskType.INITIAL_SCREENING: "gemini",
    TaskType.API_ORCHESTRATION: "gemini",

    # Claude (Paid, 20% of tasks - critical reasoning)
    TaskType.PATHOGENESIS_SYNTHESIS: "claude",
    TaskType.TARGET_PRIORITIZATION: "claude",
    TaskType.RISK_ASSESSMENT: "claude",
    TaskType.EXPERIMENT_DESIGN: "claude",
    TaskType.FINAL_INTERPRETATION: "claude",
    TaskType.REPORT_GENERATION: "claude",
    TaskType.EVIDENCE_ADJUDICATION: "claude",

    # Lobster AI (Specialized bioinformatics)
    TaskType.RNASEQ_ANALYSIS: "lobster",
    TaskType.SCRNA_ANALYSIS: "lobster",
    TaskType.GEO_DATASET_SEARCH: "lobster",
    TaskType.DIFFERENTIAL_EXPRESSION: "lobster",
    TaskType.PATHWAY_ENRICHMENT: "lobster",
    TaskType.QUALITY_CONTROL: "lobster",
}


class RoutingDecision(BaseModel):
    """Result of routing decision."""

    task_type: TaskType
    model: ModelName
    reason: str
    fallback: ModelName | None = None


class ModelRouter:
    """Routes tasks to appropriate LLM based on task type.

    Cost optimization strategy:
    - Gemini 2.5 Flash: Free tier for bulk tasks (1,500 req/day)
    - Claude Sonnet: Paid for complex reasoning (~20% of tasks)
    - Lobster AI: Specialized bioinformatics

    Usage:
        router = ModelRouter()
        decision = router.route(TaskType.PATHOGENESIS_SYNTHESIS)
        # decision.model == "claude"
    """

    def __init__(
        self,
        rules: dict[TaskType, ModelName] | None = None,
        default_model: ModelName = "gemini",
    ):
        """Initialize router with routing rules.

        Args:
            rules: Custom routing rules (uses defaults if None)
            default_model: Fallback model for unknown tasks
        """
        self.rules = rules or ROUTING_RULES
        self.default_model = default_model

    def route(self, task_type: TaskType) -> RoutingDecision:
        """Route a task to the appropriate model.

        Args:
            task_type: The type of task to route

        Returns:
            RoutingDecision with model assignment and reasoning
        """
        model = self.rules.get(task_type, self.default_model)

        # Determine reasoning
        if model == "gemini":
            reason = "Routed to Gemini (free tier) for bulk/extraction task"
            fallback = "claude"
        elif model == "claude":
            reason = "Routed to Claude (paid) for complex reasoning task"
            fallback = "gemini"
        else:  # lobster
            reason = "Routed to Lobster AI for specialized bioinformatics"
            fallback = "gemini"

        return RoutingDecision(
            task_type=task_type,
            model=model,
            reason=reason,
            fallback=fallback,
        )

    def get_model_for_task(self, task_type: TaskType) -> ModelName:
        """Simple method to get model name for a task.

        Args:
            task_type: The type of task

        Returns:
            Model name string
        """
        return self.rules.get(task_type, self.default_model)

    def estimate_cost(self, tasks: list[TaskType]) -> dict[str, int]:
        """Estimate task distribution across models.

        Args:
            tasks: List of tasks to analyze

        Returns:
            Dictionary with counts per model
        """
        counts = {"gemini": 0, "claude": 0, "lobster": 0}
        for task in tasks:
            model = self.get_model_for_task(task)
            counts[model] += 1
        return counts

    @staticmethod
    def get_gemini_tasks() -> list[TaskType]:
        """Get all tasks routed to Gemini."""
        return [t for t, m in ROUTING_RULES.items() if m == "gemini"]

    @staticmethod
    def get_claude_tasks() -> list[TaskType]:
        """Get all tasks routed to Claude."""
        return [t for t, m in ROUTING_RULES.items() if m == "claude"]

    @staticmethod
    def get_lobster_tasks() -> list[TaskType]:
        """Get all tasks routed to Lobster."""
        return [t for t, m in ROUTING_RULES.items() if m == "lobster"]
