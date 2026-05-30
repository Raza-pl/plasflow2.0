"""PyTorch device selection with MPS (Apple Silicon), CUDA, and CPU fallback.

Class constants (CLASSES, CLASS_TO_IDX, IDX_TO_CLASS) are defined here without
importing torch so they can be used in scripts and tests that don't need PyTorch.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Class labels used throughout the project
CLASSES = ["plasmid", "chromosome", "phage", "archaea"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = dict(enumerate(CLASSES))


def get_device():  # -> torch.device
    """Return the best available torch device.

    Priority: CUDA > CPU.  MPS (Apple Silicon) is intentionally skipped —
    PyTorch MPS segfaults on large float32 matrix operations (285k × 1281
    feature matrices) as of PyTorch ≤2.3.  CPU training on Apple Silicon
    still uses all performance cores and completes in ~15 min for 50 epochs.

    Set environment variable PLASFLOW_USE_MPS=1 to re-enable MPS if a future
    PyTorch version fixes the stability issues.

    Note:
        torch is imported lazily so this module can be imported without PyTorch
        in lightweight scripts (e.g. build_dataset.py, unit tests).
        MPS does not support float64. Always use .float() (float32) tensors.
    """
    import os

    import torch  # lazy import — not needed for class constants

    use_mps = os.environ.get("PLASFLOW_USE_MPS", "0") == "1"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device
