"""Inference: run classifier on sequences and return predictions with confidence.

Week 2 — Days 11–12 implementation target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from plasflow2.classify.features import extract_features
from plasflow2.classify.model import load_model
from plasflow2.utils.device import IDX_TO_CLASS, get_device

logger = logging.getLogger(__name__)

# Sequences below this confidence threshold are labelled 'unclassified'
DEFAULT_THRESHOLD = 0.7


@dataclass
class Prediction:
    """Single-sequence prediction result."""

    sequence_id: str
    label: str  # plasmid | chromosome | phage | archaea | unclassified
    confidence: float  # max softmax probability (after temperature scaling)
    scores: dict[str, float]  # per-class probabilities


def predict(
    sequences: list[str],
    sequence_ids: list[str],
    model_path: Path | str,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = 512,
) -> list[Prediction]:
    """Classify sequences using a trained MLP.

    Args:
        sequences: DNA strings.
        sequence_ids: Identifiers corresponding to each sequence.
        model_path: Path to saved .pt weights.
        threshold: Minimum confidence to assign a class label.
        batch_size: Inference batch size.

    Returns:
        List of Prediction objects, one per input sequence.

    TODO (Day 12):
        - Add temperature scaling for confidence calibration.
        - Implement MC dropout (10 passes) for uncertainty estimation.
        - Add RF ensemble member predictions.
    """
    device = get_device()
    model = load_model(model_path, device=device)

    X = extract_features(sequences)
    results: list[Prediction] = []

    for start in range(0, len(X), batch_size):
        batch = torch.tensor(X[start : start + batch_size]).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for i, prob_row in enumerate(probs):
            idx = int(np.argmax(prob_row))
            confidence = float(prob_row[idx])
            label = IDX_TO_CLASS[idx] if confidence >= threshold else "unclassified"
            results.append(
                Prediction(
                    sequence_id=sequence_ids[start + i],
                    label=label,
                    confidence=confidence,
                    scores={IDX_TO_CLASS[j]: float(prob_row[j]) for j in range(len(prob_row))},
                )
            )

    logger.info(
        "Classified %d sequences (threshold=%.2f, unclassified=%d)",
        len(results),
        threshold,
        sum(1 for r in results if r.label == "unclassified"),
    )
    return results
