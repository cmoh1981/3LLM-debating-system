"""Core module for AgingResearchAI system."""

from .schema import (
    ModuleOutput,
    Claim,
    Evidence,
    Artifact,
    NextAction,
    EvidenceTier,
    ModelType,
    ModuleStatus,
    validate_output,
    create_output,
)
from .router import ModelRouter, TaskType
from .evidence import EvidenceClassifier

__all__ = [
    "ModuleOutput",
    "Claim",
    "Evidence",
    "Artifact",
    "NextAction",
    "EvidenceTier",
    "ModelType",
    "ModuleStatus",
    "validate_output",
    "create_output",
    "ModelRouter",
    "TaskType",
    "EvidenceClassifier",
]
