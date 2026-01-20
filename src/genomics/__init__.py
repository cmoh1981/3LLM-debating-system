"""Genomics deep learning module for AgingResearchAI.

Uses Janggu for genomic sequence analysis and prediction.
"""

from .janggu_predictor import (
    JangguPredictor,
    GenomicPrediction,
    SequenceFeatures,
    create_sequence_model,
    predict_regulatory_elements,
)

__all__ = [
    "JangguPredictor",
    "GenomicPrediction",
    "SequenceFeatures",
    "create_sequence_model",
    "predict_regulatory_elements",
]
