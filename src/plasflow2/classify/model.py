"""PyTorch MLP classifier for 4-class sequence classification.

Week 2 — Day 11 implementation target.
Architecture: 2-layer MLP (input→512→128→4) with BatchNorm, GELU, Dropout.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from plasflow2.utils.device import NUM_CLASSES

logger = logging.getLogger(__name__)

INPUT_DIM = 1280  # 256 (4-mer) + 1024 (5-mer) — update to 2080 when RC k-mers added


class PlasFlowMLP(nn.Module):
    """Two-hidden-layer MLP for plasmid/chromosome/phage/archaea classification.

    Note:
        All inputs must be float32 (MPS does not support float64).
        Call model.forward(x.float()) or ensure tensors are already float32.
    """

    def __init__(self, input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())  # float32 required for MPS


def save_model(model: PlasFlowMLP, path: Path | str) -> None:
    """Save model weights to CPU (safe across platforms).

    Args:
        model: Trained PlasFlowMLP.
        path: Destination .pt file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), str(path))
    logger.info("Saved MLP weights to %s", path)


def load_model(path: Path | str, device: torch.device | None = None) -> PlasFlowMLP:
    """Load MLP weights from a .pt file.

    Args:
        path: Path to .pt file.
        device: Target device (defaults to CPU if not specified).

    Returns:
        PlasFlowMLP in eval mode.
    """
    model = PlasFlowMLP()
    state = torch.load(str(path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    if device is not None:
        model = model.to(device)
    logger.info("Loaded MLP from %s", path)
    return model
