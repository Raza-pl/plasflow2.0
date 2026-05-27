"""Unit tests for FASTA utilities.

Day 5 target: all tests pass.
"""

from plasflow2.utils.fasta import gc_content, split_by_label


def test_gc_content_pure_gc() -> None:
    assert gc_content("GGCC") == 1.0


def test_gc_content_no_gc() -> None:
    assert gc_content("AATT") == 0.0


def test_gc_content_mixed() -> None:
    result = gc_content("ACGT")
    assert abs(result - 0.5) < 1e-9


def test_gc_content_empty() -> None:
    assert gc_content("") == 0.0


def test_split_by_label_basic() -> None:
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    records = [SeqRecord(Seq("ACGT"), id=f"seq{i}") for i in range(4)]
    labels = ["plasmid", "chromosome", "plasmid", "phage"]
    bins = split_by_label(records, labels)

    assert len(bins["plasmid"]) == 2
    assert len(bins["chromosome"]) == 1
    assert len(bins["phage"]) == 1
    assert "archaea" not in bins
