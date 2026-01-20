"""Metrics collection for AgingResearchAI.

Provides in-memory metrics collection with optional export to monitoring systems.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Callable


# =============================================================================
# Metric Types
# =============================================================================

@dataclass
class MetricValue:
    """A single metric value with timestamp."""

    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""

    name: str
    count: int
    total: float
    min: float
    max: float
    avg: float
    last: float
    last_timestamp: datetime


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """Thread-safe metrics collector.

    Supports:
    - Counters: Monotonically increasing values
    - Gauges: Point-in-time values
    - Histograms: Distribution of values
    - Timers: Duration measurements
    """

    def __init__(self, retention_hours: int = 24):
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, MetricValue] = {}
        self._histograms: dict[str, list[MetricValue]] = defaultdict(list)
        self._lock = Lock()
        self._retention = timedelta(hours=retention_hours)

    # -------------------------------------------------------------------------
    # Counters
    # -------------------------------------------------------------------------

    def increment(self, name: str, value: float = 1.0, **labels):
        """Increment a counter."""
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value

    def get_counter(self, name: str, **labels) -> float:
        """Get counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0.0)

    # -------------------------------------------------------------------------
    # Gauges
    # -------------------------------------------------------------------------

    def set_gauge(self, name: str, value: float, **labels):
        """Set a gauge value."""
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = MetricValue(value=value, labels=labels)

    def get_gauge(self, name: str, **labels) -> float | None:
        """Get gauge value."""
        key = self._make_key(name, labels)
        metric = self._gauges.get(key)
        return metric.value if metric else None

    # -------------------------------------------------------------------------
    # Histograms
    # -------------------------------------------------------------------------

    def observe(self, name: str, value: float, **labels):
        """Record a value in a histogram."""
        with self._lock:
            key = self._make_key(name, labels)
            self._histograms[key].append(MetricValue(value=value, labels=labels))
            self._cleanup_old_values(key)

    def get_histogram_summary(self, name: str, **labels) -> MetricSummary | None:
        """Get histogram summary statistics."""
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])

        if not values:
            return None

        nums = [v.value for v in values]
        return MetricSummary(
            name=name,
            count=len(nums),
            total=sum(nums),
            min=min(nums),
            max=max(nums),
            avg=sum(nums) / len(nums),
            last=nums[-1],
            last_timestamp=values[-1].timestamp,
        )

    # -------------------------------------------------------------------------
    # Timers
    # -------------------------------------------------------------------------

    def timer(self, name: str, **labels) -> "Timer":
        """Create a timer context manager."""
        return Timer(self, name, labels)

    def record_duration(self, name: str, duration_ms: float, **labels):
        """Record a duration in milliseconds."""
        self.observe(f"{name}_duration_ms", duration_ms, **labels)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _make_key(self, name: str, labels: dict) -> str:
        """Create a unique key for metric + labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _cleanup_old_values(self, key: str):
        """Remove values older than retention period."""
        cutoff = datetime.utcnow() - self._retention
        self._histograms[key] = [
            v for v in self._histograms[key]
            if v.timestamp > cutoff
        ]

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all metrics as a dictionary."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": {k: v.value for k, v in self._gauges.items()},
                "histograms": {
                    k: self.get_histogram_summary(k)
                    for k in self._histograms.keys()
                },
            }

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


class Timer:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, name: str, labels: dict):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time: float | None = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            self.collector.record_duration(self.name, duration_ms, **self.labels)
        return False


# =============================================================================
# Application-Specific Metrics
# =============================================================================

class AgingResearchMetrics:
    """Pre-defined metrics for AgingResearchAI."""

    def __init__(self, collector: MetricsCollector | None = None):
        self.collector = collector or MetricsCollector()

    # -------------------------------------------------------------------------
    # API Metrics
    # -------------------------------------------------------------------------

    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
    ):
        """Record an API request."""
        self.collector.increment(
            "api_requests_total",
            endpoint=endpoint,
            method=method,
            status_code=str(status_code),
        )
        self.collector.observe(
            "api_request_duration_ms",
            duration_ms,
            endpoint=endpoint,
        )

        if status_code >= 500:
            self.collector.increment("api_errors_total", endpoint=endpoint)

    # -------------------------------------------------------------------------
    # LLM Metrics
    # -------------------------------------------------------------------------

    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        success: bool = True,
    ):
        """Record an LLM API call."""
        self.collector.increment("llm_calls_total", model=model)
        self.collector.increment("llm_input_tokens_total", input_tokens, model=model)
        self.collector.increment("llm_output_tokens_total", output_tokens, model=model)
        self.collector.observe("llm_call_duration_ms", duration_ms, model=model)

        if not success:
            self.collector.increment("llm_errors_total", model=model)

    def record_llm_cost(self, model: str, cost_usd: float):
        """Record LLM cost."""
        self.collector.increment("llm_cost_usd_total", cost_usd, model=model)

    # -------------------------------------------------------------------------
    # Debate Metrics
    # -------------------------------------------------------------------------

    def record_debate(
        self,
        rounds: int,
        consensus_count: int,
        rejected_count: int,
        unresolved_count: int,
        duration_ms: float,
    ):
        """Record a debate session."""
        self.collector.increment("debates_total")
        self.collector.increment("debate_rounds_total", rounds)
        self.collector.increment("debate_consensus_claims_total", consensus_count)
        self.collector.increment("debate_rejected_claims_total", rejected_count)
        self.collector.increment("debate_unresolved_claims_total", unresolved_count)
        self.collector.observe("debate_duration_ms", duration_ms)

    # -------------------------------------------------------------------------
    # RAG Metrics
    # -------------------------------------------------------------------------

    def record_rag_search(
        self,
        collection: str,
        results_count: int,
        duration_ms: float,
    ):
        """Record a RAG search."""
        self.collector.increment("rag_searches_total", collection=collection)
        self.collector.observe("rag_results_count", results_count, collection=collection)
        self.collector.observe("rag_search_duration_ms", duration_ms, collection=collection)

    def record_rag_ingest(
        self,
        source: str,
        collection: str,
        documents_count: int,
        duration_ms: float,
    ):
        """Record document ingestion."""
        self.collector.increment("rag_ingests_total", source=source, collection=collection)
        self.collector.increment("rag_documents_ingested_total", documents_count, source=source)
        self.collector.observe("rag_ingest_duration_ms", duration_ms, source=source)

    # -------------------------------------------------------------------------
    # ADMET Metrics
    # -------------------------------------------------------------------------

    def record_admet_prediction(
        self,
        risk_level: str,
        duration_ms: float,
        success: bool = True,
    ):
        """Record an ADMET prediction."""
        self.collector.increment("admet_predictions_total")
        self.collector.increment(f"admet_risk_{risk_level.lower()}_total")
        self.collector.observe("admet_prediction_duration_ms", duration_ms)

        if not success:
            self.collector.increment("admet_errors_total")

    # -------------------------------------------------------------------------
    # Pipeline Metrics
    # -------------------------------------------------------------------------

    def record_pipeline_run(
        self,
        workflow: str,
        disease: str,
        claims_count: int,
        duration_ms: float,
        success: bool = True,
    ):
        """Record a pipeline run."""
        self.collector.increment("pipeline_runs_total", workflow=workflow, disease=disease)
        self.collector.observe("pipeline_claims_count", claims_count, workflow=workflow)
        self.collector.observe("pipeline_duration_ms", duration_ms, workflow=workflow)

        if not success:
            self.collector.increment("pipeline_errors_total", workflow=workflow)

    # -------------------------------------------------------------------------
    # System Metrics
    # -------------------------------------------------------------------------

    def set_service_status(self, service: str, is_healthy: bool):
        """Set service health status."""
        self.collector.set_gauge(
            "service_health",
            1.0 if is_healthy else 0.0,
            service=service,
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Get metrics summary for dashboard."""
        metrics = self.collector.get_all_metrics()
        counters = metrics.get("counters", {})

        return {
            "api": {
                "total_requests": counters.get("api_requests_total", 0),
                "total_errors": counters.get("api_errors_total", 0),
            },
            "llm": {
                "total_calls": sum(
                    v for k, v in counters.items()
                    if k.startswith("llm_calls_total")
                ),
                "total_input_tokens": sum(
                    v for k, v in counters.items()
                    if k.startswith("llm_input_tokens_total")
                ),
                "total_output_tokens": sum(
                    v for k, v in counters.items()
                    if k.startswith("llm_output_tokens_total")
                ),
                "total_cost_usd": sum(
                    v for k, v in counters.items()
                    if k.startswith("llm_cost_usd_total")
                ),
            },
            "debate": {
                "total_debates": counters.get("debates_total", 0),
                "consensus_claims": counters.get("debate_consensus_claims_total", 0),
                "rejected_claims": counters.get("debate_rejected_claims_total", 0),
            },
            "rag": {
                "total_searches": sum(
                    v for k, v in counters.items()
                    if k.startswith("rag_searches_total")
                ),
                "documents_ingested": sum(
                    v for k, v in counters.items()
                    if k.startswith("rag_documents_ingested_total")
                ),
            },
            "admet": {
                "total_predictions": counters.get("admet_predictions_total", 0),
            },
            "pipeline": {
                "total_runs": sum(
                    v for k, v in counters.items()
                    if k.startswith("pipeline_runs_total")
                ),
            },
        }


# =============================================================================
# Global Metrics Instance
# =============================================================================

_metrics: AgingResearchMetrics | None = None


def get_metrics() -> AgingResearchMetrics:
    """Get or create global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = AgingResearchMetrics()
    return _metrics
