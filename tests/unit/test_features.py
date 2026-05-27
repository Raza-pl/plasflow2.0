"""Unit tests for k-mer feature extraction.

Day 9 target: all tests pass.
"""

import numpy as np
from plasflow2.classify.features import (
    FEATURE_DIM,
    extract_features,
    kmer_vector,
)


def test_kmer_vector_shape() -> None:
    seq = "ACGTACGTACGT"
    v = kmer_vector(seq, k=4)
    assert v.shape == (256,), f"Expected (256,) got {v.shape}"


def test_kmer_vector_normalised() -> None:
    seq = "ACGTACGTACGT" * 10
    v = kmer_vector(seq, k=4)
    norm = float(np.linalg.norm(v))
    assert abs(norm - 1.0) < 1e-5, f"Expected L2-norm ≈ 1.0, got {norm}"


def test_kmer_vector_short_sequence() -> None:
    """Sequences shorter than k should produce a zero vector."""
    v = kmer_vector("ACG", k=4)
    assert np.all(v == 0)


def test_extract_features_shape() -> None:
    seqs = ["ACGT" * 100, "GGCC" * 100, "TTAA" * 100]
    X = extract_features(seqs)
    assert X.shape == (3, FEATURE_DIM), f"Expected (3, {FEATURE_DIM}), got {X.shape}"


def test_extract_features_dtype() -> None:
    seqs = ["ACGT" * 50]
    X = extract_features(seqs)
    assert X.dtype == np.float32


def test_extract_features_different_seqs() -> None:
    """Different sequences should produce different feature vectors."""
    X = extract_features(["ACGT" * 100, "GGCC" * 100])
    assert not np.allclose(X[0], X[1]), "Expected distinct vectors for distinct sequences"


def test_kmer_vector_rc_invariant() -> None:
    """Reverse complement of a sequence should produce the same feature vector."""
    from plasflow2.classify.features import _reverse_complement

    seq = "ACGTTAGCCA" * 20
    rc = _reverse_complement(seq)
    v_fwd = kmer_vector(seq, k=4)
    v_rc = kmer_vector(rc, k=4)
    np.testing.assert_allclose(
        v_fwd, v_rc, atol=1e-5, err_msg="RC of sequence should yield identical k-mer vector"
    )


def test_reverse_complement_correctness() -> None:
    """Spot-check the RC helper."""
    from plasflow2.classify.features import _reverse_complement

    assert _reverse_complement("ACGT") == "ACGT"  # palindrome
    assert _reverse_complement("AAAA") == "TTTT"
    assert _reverse_complement("GCGC") == "GCGC"  # palindrome
    assert _reverse_complement("ATCG") == "CGAT"
