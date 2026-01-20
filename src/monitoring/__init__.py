"""Monitoring, logging, and cost tracking for AgingResearchAI.

Provides:
- Centralized logging with structured output
- Metrics collection and aggregation
- LLM cost tracking and budget management
- Audit logging for compliance
"""

from .logger import (
    setup_logging,
    get_logger,
    LogContext,
    log_with_context,
    AuditLogger,
    ensure_logging,
)
from .metrics import (
    MetricsCollector,
    MetricSummary,
    Timer,
    AgingResearchMetrics,
    get_metrics,
)
from .cost_tracker import (
    CostTracker,
    UsageRecord,
    DailySummary,
    get_cost_tracker,
    quick_cost_estimate,
    LLM_PRICING,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "LogContext",
    "log_with_context",
    "AuditLogger",
    "ensure_logging",
    # Metrics
    "MetricsCollector",
    "MetricSummary",
    "Timer",
    "AgingResearchMetrics",
    "get_metrics",
    # Cost Tracking
    "CostTracker",
    "UsageRecord",
    "DailySummary",
    "get_cost_tracker",
    "quick_cost_estimate",
    "LLM_PRICING",
]
