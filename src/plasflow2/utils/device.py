"""PyTorch device selection with MPS (Apple Silicon), CUDA, and CPU fallback."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Class labels used throughout the project
CLASSES = ["plasmid", "chromosome", "phage", "archaea"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = dict(enumerate(CLASSES))


def get_device() -> torch.device:
    """Return the best available torch device.

    Priority: MPS (Apple Silicon) > CUDA > CPU.

    Note:
        MPS does not support float64. Always use .float() (float32) tensors.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device
