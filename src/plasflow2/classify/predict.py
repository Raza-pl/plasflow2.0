"""Inference: run classifier on sequences and return predictions with confidence.

Week 2 — Days 11–12 implementation target.

Class-specific thresholds
-------------------------
The MLP is trained on a balanced dataset (~25 % per class), but real
metagenome assemblies contain only ~2–5 % plasmid contigs.  Using a single
confidence threshold therefore overestimates plasmid prevalence because the
model has never learned that plasmid is a rare class.

To correct for this prior imbalance we apply *class-specific* thresholds:

* **plasmid** — default 0.95 (high bar; false positives are costly).
* **chromosome / phage / archaea** — default 0.70 (lower bar; these are
  abundant and the cost of a missed call is lower).

Users can override these via the CLI flags ``--threshold`` (all non-plasmid
classes) and ``--plasmid-threshold`` (plasmid only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plasflow2.classify.features import extract_features
from plasflow2.utils.device import IDX_TO_CLASS, get_device

logger = logging.getLogger(__name__)

# Default confidence thresholds (class-specific)
DEFAULT_THRESHOLD = 0.70  # chromosome / phage / archaea
DEFAULT_PLASMID_THRESHOLD = 0.95  # plasmid — higher bar to correct for class-prior imbalance


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
    plasmid_threshold: float = DEFAULT_PLASMID_THRESHOLD,
    batch_size: int = 512,
) -> list[Prediction]:
    """Classify sequences using a trained MLP with class-specific thresholds.

    For each sequence the model's argmax class is selected, then a
    *class-specific* confidence threshold is applied:

    * ``plasmid_threshold`` governs plasmid calls (default 0.95).
    * ``threshold`` governs all other classes (default 0.70).

    Sequences whose winning class falls below the applicable threshold are
    labelled ``unclassified``.

    Args:
        sequences: DNA strings.
        sequence_ids: Identifiers corresponding to each sequence.
        model_path: Path to saved .pt weights.
        threshold: Minimum confidence for chromosome / phage / archaea calls.
        plasmid_threshold: Minimum confidence for plasmid calls (higher than
            ``threshold`` to compensate for class-prior imbalance).
        batch_size: Inference batch size.

    Returns:
        List of Prediction objects, one per input sequence.
    """
    import torch

    from plasflow2.classify.model import load_model

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
            best_class = IDX_TO_CLASS[idx]
            # Apply class-specific threshold: plasmid requires higher confidence
            # to compensate for class-prior imbalance (model trained ~25% plasmid
            # but real metagenomes have ~2–5% plasmid).
            applicable_threshold = plasmid_threshold if best_class == "plasmid" else threshold
            label = best_class if confidence >= applicable_threshold else "unclassified"
            results.append(
                Prediction(
                    sequence_id=sequence_ids[start + i],
                    label=label,
                    confidence=confidence,
                    scores={IDX_TO_CLASS[j]: float(prob_row[j]) for j in range(len(prob_row))},
                )
            )

    logger.info(
        "Classified %d sequences (threshold=%.2f, plasmid_threshold=%.2f, unclassified=%d)",
        len(results),
        threshold,
        plasmid_threshold,
        sum(1 for r in results if r.label == "unclassified"),
    )
    return results
