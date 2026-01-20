"""Cost tracking for LLM API usage in AgingResearchAI.

Tracks token usage and calculates costs for different LLM providers.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any


# =============================================================================
# Pricing Configuration
# =============================================================================

# Prices per 1 million tokens (as of 2026-01)
LLM_PRICING = {
    "gemini": {
        "input": 0.0,      # Free tier
        "output": 0.0,
        "notes": "Free tier (1,500 req/day limit)",
    },
    "gemini-2.5-flash": {
        "input": 0.0,
        "output": 0.0,
        "notes": "Free tier",
    },
    "grok": {
        "input": 5.0,      # Grok-beta
        "output": 15.0,
        "notes": "Grok (xAI)",
    },
    "grok-beta": {
        "input": 5.0,
        "output": 15.0,
        "notes": "Grok Beta",
    },
    "grok-2": {
        "input": 2.0,
        "output": 10.0,
        "notes": "Grok 2",
    },
    "deepseek": {
        "input": 0.14,     # DeepSeek V3.2
        "output": 0.28,
        "notes": "DeepSeek V3.2",
    },
    "deepseek-v3": {
        "input": 0.14,
        "output": 0.28,
        "notes": "DeepSeek V3",
    },
    "kimi": {
        "input": 0.20,     # Kimi K2
        "output": 0.40,
        "notes": "Kimi K2 / Moonshot",
    },
    "kimi-k2": {
        "input": 0.20,
        "output": 0.40,
        "notes": "Kimi K2",
    },
    "claude": {
        "input": 3.00,     # Claude Sonnet
        "output": 15.00,
        "notes": "Claude Sonnet 4",
    },
    "claude-sonnet": {
        "input": 3.00,
        "output": 15.00,
        "notes": "Claude Sonnet 4",
    },
    "claude-opus": {
        "input": 15.00,
        "output": 75.00,
        "notes": "Claude Opus 4.5",
    },
    "openai": {
        "input": 2.50,     # GPT-4o
        "output": 10.00,
        "notes": "GPT-4o",
    },
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
        "notes": "GPT-4o",
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
        "notes": "GPT-4o Mini",
    },
}


# =============================================================================
# Usage Tracking
# =============================================================================

@dataclass
class UsageRecord:
    """Record of a single API call."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    operation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DailySummary:
    """Daily usage summary."""

    date: str
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    by_model: dict[str, dict[str, float]]


class CostTracker:
    """Track and calculate LLM API costs.

    Features:
    - Real-time cost calculation
    - Daily/monthly summaries
    - Budget alerts
    - Cost optimization recommendations
    """

    def __init__(
        self,
        daily_budget_usd: float = 10.0,
        monthly_budget_usd: float = 100.0,
        retention_days: int = 30,
    ):
        self.daily_budget = daily_budget_usd
        self.monthly_budget = monthly_budget_usd
        self.retention_days = retention_days

        self._records: list[UsageRecord] = []
        self._lock = Lock()

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "",
        **metadata,
    ) -> UsageRecord:
        """Record API usage and calculate cost.

        Args:
            model: Model name (gemini, deepseek, kimi, claude, openai)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            operation: Type of operation (debate, rag, pipeline)
            **metadata: Additional metadata

        Returns:
            UsageRecord with calculated cost
        """
        # Normalize model name
        model_key = model.lower().replace("-", "_").split("_")[0]

        # Get pricing
        pricing = LLM_PRICING.get(model_key, LLM_PRICING.get(model.lower(), {
            "input": 0.0,
            "output": 0.0,
        }))

        # Calculate cost (per million tokens)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        record = UsageRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(total_cost, 6),
            operation=operation,
            metadata=metadata,
        )

        with self._lock:
            self._records.append(record)
            self._cleanup_old_records()

        return record

    def _cleanup_old_records(self):
        """Remove records older than retention period."""
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        self._records = [r for r in self._records if r.timestamp > cutoff]

    def get_daily_summary(self, date: datetime | None = None) -> DailySummary:
        """Get usage summary for a specific day.

        Args:
            date: Date to summarize (defaults to today)

        Returns:
            DailySummary for the specified day
        """
        if date is None:
            date = datetime.utcnow()

        date_str = date.strftime("%Y-%m-%d")
        day_start = datetime.strptime(date_str, "%Y-%m-%d")
        day_end = day_start + timedelta(days=1)

        with self._lock:
            day_records = [
                r for r in self._records
                if day_start <= r.timestamp < day_end
            ]

        # Aggregate by model
        by_model: dict[str, dict[str, float]] = {}
        for record in day_records:
            if record.model not in by_model:
                by_model[record.model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
            by_model[record.model]["calls"] += 1
            by_model[record.model]["input_tokens"] += record.input_tokens
            by_model[record.model]["output_tokens"] += record.output_tokens
            by_model[record.model]["cost_usd"] += record.cost_usd

        return DailySummary(
            date=date_str,
            total_calls=len(day_records),
            total_input_tokens=sum(r.input_tokens for r in day_records),
            total_output_tokens=sum(r.output_tokens for r in day_records),
            total_cost_usd=round(sum(r.cost_usd for r in day_records), 4),
            by_model=by_model,
        )

    def get_monthly_summary(self, year: int | None = None, month: int | None = None) -> dict:
        """Get usage summary for a specific month."""
        now = datetime.utcnow()
        if year is None:
            year = now.year
        if month is None:
            month = now.month

        month_start = datetime(year, month, 1)
        if month == 12:
            month_end = datetime(year + 1, 1, 1)
        else:
            month_end = datetime(year, month + 1, 1)

        with self._lock:
            month_records = [
                r for r in self._records
                if month_start <= r.timestamp < month_end
            ]

        # Daily breakdown
        daily_costs = {}
        for record in month_records:
            day = record.timestamp.strftime("%Y-%m-%d")
            if day not in daily_costs:
                daily_costs[day] = 0.0
            daily_costs[day] += record.cost_usd

        return {
            "year": year,
            "month": month,
            "total_calls": len(month_records),
            "total_cost_usd": round(sum(r.cost_usd for r in month_records), 4),
            "daily_costs": daily_costs,
            "budget_usd": self.monthly_budget,
            "budget_remaining": round(
                self.monthly_budget - sum(r.cost_usd for r in month_records), 4
            ),
        }

    def check_budget(self) -> dict:
        """Check current budget status.

        Returns:
            Dictionary with budget status and alerts
        """
        daily = self.get_daily_summary()
        monthly = self.get_monthly_summary()

        alerts = []

        # Daily budget check
        daily_pct = (daily.total_cost_usd / self.daily_budget) * 100 if self.daily_budget > 0 else 0
        if daily_pct >= 100:
            alerts.append({
                "level": "critical",
                "message": f"Daily budget exceeded: ${daily.total_cost_usd:.2f} / ${self.daily_budget:.2f}",
            })
        elif daily_pct >= 80:
            alerts.append({
                "level": "warning",
                "message": f"Daily budget at {daily_pct:.0f}%: ${daily.total_cost_usd:.2f} / ${self.daily_budget:.2f}",
            })

        # Monthly budget check
        monthly_pct = (monthly["total_cost_usd"] / self.monthly_budget) * 100 if self.monthly_budget > 0 else 0
        if monthly_pct >= 100:
            alerts.append({
                "level": "critical",
                "message": f"Monthly budget exceeded: ${monthly['total_cost_usd']:.2f} / ${self.monthly_budget:.2f}",
            })
        elif monthly_pct >= 80:
            alerts.append({
                "level": "warning",
                "message": f"Monthly budget at {monthly_pct:.0f}%",
            })

        return {
            "daily": {
                "spent_usd": daily.total_cost_usd,
                "budget_usd": self.daily_budget,
                "percentage": round(daily_pct, 1),
            },
            "monthly": {
                "spent_usd": monthly["total_cost_usd"],
                "budget_usd": self.monthly_budget,
                "percentage": round(monthly_pct, 1),
            },
            "alerts": alerts,
            "is_within_budget": len([a for a in alerts if a["level"] == "critical"]) == 0,
        }

    def get_cost_optimization_tips(self) -> list[str]:
        """Get cost optimization recommendations.

        Returns:
            List of optimization tips based on usage patterns
        """
        tips = []
        daily = self.get_daily_summary()

        # Analyze model usage
        if daily.by_model:
            # Check if expensive models are being used heavily
            expensive_models = ["claude", "openai", "gpt-4"]
            for model, usage in daily.by_model.items():
                if any(exp in model.lower() for exp in expensive_models):
                    if usage["calls"] > 10:
                        tips.append(
                            f"High usage of {model} ({usage['calls']} calls). "
                            "Consider using DeepSeek or Kimi for critique tasks."
                        )

            # Check if Gemini is being underutilized
            if "gemini" not in daily.by_model and daily.total_calls > 0:
                tips.append(
                    "Gemini (free tier) not being used. "
                    "Route proposal tasks to Gemini to reduce costs."
                )

        # Token efficiency
        if daily.total_input_tokens > 0:
            output_ratio = daily.total_output_tokens / daily.total_input_tokens
            if output_ratio < 0.1:
                tips.append(
                    "Low output/input ratio. Consider using more specific prompts "
                    "to reduce input tokens."
                )

        # General tips
        if not tips:
            tips.append("Usage patterns look optimal. Keep using the 3-LLM debate system!")

        return tips

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost for a planned API call.

        Args:
            model: Model name
            input_tokens: Expected input tokens
            output_tokens: Expected output tokens

        Returns:
            Estimated cost in USD
        """
        model_key = model.lower()
        pricing = LLM_PRICING.get(model_key, {"input": 0.0, "output": 0.0})

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return round(input_cost + output_cost, 6)

    def get_pricing_info(self) -> dict:
        """Get current pricing information."""
        return {
            "pricing_per_million_tokens": LLM_PRICING,
            "recommended_config": {
                "proposer": "gemini (FREE)",
                "critic": "deepseek (~$0.14/M)",
                "judge": "kimi (~$0.20/M)",
            },
            "estimated_daily_cost": {
                "light_usage": "$0.50 - $1.00",
                "moderate_usage": "$1.00 - $3.00",
                "heavy_usage": "$3.00 - $5.00",
            },
        }


# =============================================================================
# Global Cost Tracker
# =============================================================================

_cost_tracker: CostTracker | None = None


def get_cost_tracker(
    daily_budget_usd: float = 10.0,
    monthly_budget_usd: float = 100.0,
) -> CostTracker:
    """Get or create global cost tracker."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker(
            daily_budget_usd=daily_budget_usd,
            monthly_budget_usd=monthly_budget_usd,
        )
    return _cost_tracker


def quick_cost_estimate(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    """Quick cost estimate without tracking.

    Args:
        model: Model name
        input_tokens: Input token count
        output_tokens: Output token count

    Returns:
        Cost breakdown dictionary
    """
    model_key = model.lower()
    pricing = LLM_PRICING.get(model_key, {"input": 0.0, "output": 0.0, "notes": "Unknown model"})

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
        "pricing_notes": pricing.get("notes", ""),
    }
