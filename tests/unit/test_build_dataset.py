"""Unit tests for build_dataset.py helpers.

Tests use in-memory synthetic FASTAs so no real database files are needed.
"""

from __future__ import annotations

import gzip

# We import the helpers directly from the script; add scripts/ to sys.path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from build_dataset import fragment_sequences, load_and_subsample  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SEQ = "ACGT" * 500  # 2000 bp — passes min_length=1000


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    """Write (id, sequence) pairs to a plain FASTA file."""
    with open(path, "w") as fh:
        for rid, seq in records:
            fh.write(f">{rid}\n{seq}\n")


def _write_fasta_gz(path: Path, records: list[tuple[str, str]]) -> None:
    """Write (id, sequence) pairs to a gzipped FASTA file."""
    with gzip.open(path, "wt") as fh:
        for rid, seq in records:
            fh.write(f">{rid}\n{seq}\n")


# ---------------------------------------------------------------------------
# load_and_subsample — plain FASTA
# ---------------------------------------------------------------------------


def test_load_plain_fasta_counts(tmp_path: Path) -> None:
    fa = tmp_path / "test.fna"
    _write_fasta(fa, [(f"seq{i}", _SEQ) for i in range(10)])
    seqs, ids, labels = load_and_subsample(fa, "plasmid", max_per_class=20)
    assert len(seqs) == 10
    assert len(ids) == 10
    assert len(labels) == 10


def test_load_gzipped_fasta(tmp_path: Path) -> None:
    fa = tmp_path / "test.fa.gz"
    _write_fasta_gz(fa, [(f"seq{i}", _SEQ) for i in range(5)])
    seqs, ids, labels = load_and_subsample(fa, "phage", max_per_class=20)
    assert len(seqs) == 5


def test_subsample_cap(tmp_path: Path) -> None:
    """When there are more sequences than cap, only cap are returned."""
    fa = tmp_path / "test.fna"
    _write_fasta(fa, [(f"seq{i}", _SEQ) for i in range(50)])
    seqs, ids, labels = load_and_subsample(fa, "plasmid", max_per_class=10)
    assert len(seqs) == 10


def test_min_length_filter(tmp_path: Path) -> None:
    """Sequences shorter than min_length should be excluded."""
    fa = tmp_path / "test.fna"
    short = "ACGT" * 10  # 40 bp — below default min_length=1000
    _write_fasta(fa, [("short", short), ("long", _SEQ)])
    seqs, ids, labels = load_and_subsample(fa, "chromosome", max_per_class=100, min_length=1000)
    assert len(seqs) == 1
    assert ids[0] == "long"


def test_labels_are_correct_class_index(tmp_path: Path) -> None:
    """Label values should match CLASS_TO_IDX for the given class name."""
    from plasflow2.utils.device import CLASS_TO_IDX

    fa = tmp_path / "test.fna"
    _write_fasta(fa, [("s1", _SEQ), ("s2", _SEQ)])
    for class_name in ("plasmid", "chromosome", "phage", "archaea"):
        _, _, labels = load_and_subsample(fa, class_name, max_per_class=10)
        expected_idx = CLASS_TO_IDX[class_name]
        assert all(
            lbl == expected_idx for lbl in labels
        ), f"Wrong label index for {class_name}: got {labels[0]}, expected {expected_idx}"


def test_no_duplicate_ids_after_subsample(tmp_path: Path) -> None:
    fa = tmp_path / "test.fna"
    _write_fasta(fa, [(f"seq{i}", _SEQ) for i in range(30)])
    _, ids, _ = load_and_subsample(fa, "plasmid", max_per_class=15)
    assert len(ids) == len(set(ids)), "Duplicate sequence IDs after subsampling"


def test_subsample_reproducible(tmp_path: Path) -> None:
    """Same seed → same subsample."""
    fa = tmp_path / "test.fna"
    _write_fasta(fa, [(f"seq{i}", _SEQ) for i in range(50)])
    _, ids1, _ = load_and_subsample(fa, "plasmid", max_per_class=10, seed=42)
    _, ids2, _ = load_and_subsample(fa, "plasmid", max_per_class=10, seed=42)
    assert ids1 == ids2


def test_subsample_different_seeds(tmp_path: Path) -> None:
    """Different seeds → different subsample (with overwhelming probability)."""
    fa = tmp_path / "test.fna"
    _write_fasta(fa, [(f"seq{i}", _SEQ) for i in range(50)])
    _, ids1, _ = load_and_subsample(fa, "plasmid", max_per_class=10, seed=1)
    _, ids2, _ = load_and_subsample(fa, "plasmid", max_per_class=10, seed=99)
    assert ids1 != ids2


# ---------------------------------------------------------------------------
# fragment_sequences
# ---------------------------------------------------------------------------

_CHROM = "ACGT" * 5000  # 20 000 bp synthetic chromosome


def test_fragment_produces_correct_window_sizes() -> None:
    frags, ids = fragment_sequences([_CHROM], ["chr1"], window_sizes=(1000, 2000))
    for frag in frags:
        assert len(frag) in (1000, 2000), f"Unexpected fragment length {len(frag)}"


def test_fragment_ids_encode_parent_and_position() -> None:
    frags, ids = fragment_sequences([_CHROM], ["chrA"], window_sizes=(1000,), step_fraction=1.0)
    for fid in ids:
        assert fid.startswith("chrA_w1000_s"), f"Unexpected ID format: {fid}"


def test_fragment_count_matches_window_math() -> None:
    """Non-overlapping (step_fraction=1.0) windows: expect floor((L-w)/w)+1 windows."""
    seq = "A" * 10_000
    frags, _ = fragment_sequences([seq], ["s"], window_sizes=(1000,), step_fraction=1.0)
    expected = 10  # 10 000 / 1000 = 10
    assert len(frags) == expected, f"Expected {expected} fragments, got {len(frags)}"


def test_fragment_max_fragments_cap() -> None:
    frags, ids = fragment_sequences(
        [_CHROM], ["chr1"], window_sizes=(1000, 2000, 5000), max_fragments=50
    )
    assert len(frags) == 50
    assert len(ids) == 50


def test_fragment_short_seq_skipped() -> None:
    """Sequences shorter than the window size should produce no fragments."""
    short = "ACGT" * 100  # 400 bp
    frags, _ = fragment_sequences([short], ["s"], window_sizes=(1000,))
    assert len(frags) == 0


def test_fragment_reproducible() -> None:
    f1, i1 = fragment_sequences([_CHROM], ["c"], window_sizes=(2000,), max_fragments=20, seed=7)
    f2, i2 = fragment_sequences([_CHROM], ["c"], window_sizes=(2000,), max_fragments=20, seed=7)
    assert i1 == i2


def test_non_acgt_sequences_excluded(tmp_path: Path) -> None:
    """Sequences containing non-ACGTN characters should be filtered out."""
    fa = tmp_path / "test.fna"
    bad_seq = "ACGTRYSWKM" * 200  # contains IUPAC ambiguity codes
    _write_fasta(fa, [("bad", bad_seq), ("good", _SEQ)])
    seqs, ids, _ = load_and_subsample(fa, "plasmid", max_per_class=10)
    assert "bad" not in ids
    assert "good" in ids
