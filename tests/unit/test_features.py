"""Unit tests for k-mer feature extraction.

FEATURE_DIM = 1281: 256 (4-mer) + 1024 (5-mer) + 1 (log10 length).
"""

import numpy as np
from plasflow2.classify.features import (
    _KMER_DIM,
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


def test_feature_dim_is_1281() -> None:
    """FEATURE_DIM must equal _KMER_DIM + 1 (length feature)."""
    assert FEATURE_DIM == _KMER_DIM + 1
    assert FEATURE_DIM == 1281


def test_length_feature_increases_with_sequence_length() -> None:
    """Longer sequences should have a larger length feature (last column)."""
    short = "ACGT" * 250  # 1000 bp
    long = "ACGT" * 2500  # 10000 bp
    X = extract_features([short, long])
    assert X[1, -1] > X[0, -1], "Length feature should be larger for the longer sequence"


def test_length_feature_in_zero_one_range() -> None:
    """Length feature (last column) must be in [0, 1] for typical contig lengths."""
    seqs = ["ACGT" * 250, "ACGT" * 2500, "ACGT" * 25000]  # 1 kb, 10 kb, 100 kb
    X = extract_features(seqs)
    assert np.all(X[:, -1] >= 0.0)
    assert np.all(X[:, -1] <= 1.0)
