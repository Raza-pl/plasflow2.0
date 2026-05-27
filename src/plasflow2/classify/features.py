"""k-mer frequency feature extraction.

Week 2 — Day 9 implementation target.
Day 3 update: reverse-complement aware counting (strand-invariant features).

Feature vector: k=4 (256 dims) + k=5 (1024 dims) = 1280 dims, L2-normalised.
Counts from both the forward strand and its reverse complement are merged before
normalisation, making features strand-invariant (required for double-stranded DNA).
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# k-mer sizes to use
KMER_SIZES = (4, 5)

# Complement mapping for reverse-complement calculation
_COMPLEMENT: dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C"}


def _all_kmers(k: int) -> list[str]:
    """Return sorted list of all k-mers over {A, C, G, T}."""
    return ["".join(p) for p in itertools.product("ACGT", repeat=k)]


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA string.

    Non-ACGT characters are dropped (treated as absent).
    """
    return "".join(_COMPLEMENT.get(b, "") for b in reversed(seq.upper()))


# Pre-build k-mer vocabularies and index maps
_VOCAB: dict[int, list[str]] = {k: _all_kmers(k) for k in KMER_SIZES}
_KMER_TO_IDX: dict[int, dict[str, int]] = {
    k: {km: i for i, km in enumerate(vocab)} for k, vocab in _VOCAB.items()
}

FEATURE_DIM = sum(len(_VOCAB[k]) for k in KMER_SIZES)  # 256 + 1024 = 1280


def kmer_vector(seq: str, k: int) -> NDArray[np.float32]:
    """Compute normalised k-mer frequency vector for one sequence.

    Counts k-mers from both the forward strand and its reverse complement,
    then merges them into a single strand-invariant frequency vector before
    L2-normalisation.

    Args:
        seq: DNA string (any case; non-ACGT characters are skipped).
        k: k-mer size (must be a key in KMER_SIZES).

    Returns:
        Float32 array of shape (4**k,), L2-normalised.
        Returns a zero vector if the sequence is shorter than k.
    """
    vocab_size = 4**k
    idx_map = _KMER_TO_IDX[k]
    counts = np.zeros(vocab_size, dtype=np.float32)
    seq = seq.upper()
    rc_seq = _reverse_complement(seq)

    for strand in (seq, rc_seq):
        for i in range(len(strand) - k + 1):
            kmer = strand[i : i + k]
            if kmer in idx_map:
                counts[idx_map[kmer]] += 1

    norm = np.linalg.norm(counts)
    if norm > 0:
        counts /= norm
    return counts


def extract_features(sequences: list[str]) -> NDArray[np.float32]:
    """Extract concatenated k-mer feature matrix for a list of sequences.

    Args:
        sequences: List of DNA strings.

    Returns:
        Float32 array of shape (N, FEATURE_DIM).
    """
    n = len(sequences)
    X = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    offset = 0
    for k in KMER_SIZES:
        dim = 4**k
        for i, seq in enumerate(sequences):
            X[i, offset : offset + dim] = kmer_vector(seq, k)
        offset += dim
    logger.info("Extracted features: shape %s", X.shape)
    return X


def save_features(X: NDArray[np.float32], path: Path | str) -> None:
    """Save feature matrix to an .npy file.

    TODO (Day 9): Switch to HDF5 (h5py) for datasets > 1 GB.
    """
    np.save(str(path), X)


def load_features(path: Path | str) -> NDArray[np.float32]:
    """Load feature matrix from an .npy file."""
    return np.load(str(path)).astype(np.float32)
