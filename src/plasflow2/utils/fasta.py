"""FASTA parsing, filtering, and writing utilities.

Week 1 — Day 5 implementation target.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

from Bio import SeqIO  # type: ignore[import]
from Bio.SeqRecord import SeqRecord  # type: ignore[import]

logger = logging.getLogger(__name__)


def load_fasta(path: Path | str, min_length: int = 1000) -> list[SeqRecord]:
    """Load sequences from a FASTA file, filtering by minimum length.

    Args:
        path: Path to FASTA file.
        min_length: Minimum sequence length to keep (default 1000 bp).

    Returns:
        List of SeqRecord objects passing the length filter.

    TODO (Day 5):
        - Implement GC% computation and attach to SeqRecord.letter_annotations
        - Validate sequence alphabet (DNA only)
        - Handle gzipped FASTA (.fa.gz, .fasta.gz)
    """
    records: list[SeqRecord] = []
    total = 0
    for record in SeqIO.parse(str(path), "fasta"):
        total += 1
        if len(record.seq) >= min_length:
            records.append(record)
    logger.info(
        "Loaded %d/%d sequences from %s (min_length=%d)",
        len(records),
        total,
        path,
        min_length,
    )
    return records


def gc_content(seq: str) -> float:
    """Compute GC content (0–1) for a DNA string."""
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq) if seq else 0.0


def write_fasta(records: list[SeqRecord], path: Path | str) -> None:
    """Write SeqRecord list to a FASTA file.

    Args:
        records: Sequences to write.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, str(path), "fasta")
    logger.info("Wrote %d sequences to %s", len(records), path)


def iter_fasta(path: Path | str) -> Generator[SeqRecord, None, None]:
    """Lazily iterate over sequences in a FASTA file (memory-efficient)."""
    yield from SeqIO.parse(str(path), "fasta")


def split_by_label(
    records: list[SeqRecord],
    labels: list[str],
) -> dict[str, list[SeqRecord]]:
    """Bin sequences into groups by their predicted label.

    Args:
        records: Sequences (same order as labels).
        labels: Predicted class per sequence.

    Returns:
        Dict mapping class name → list of SeqRecord.

    TODO (Week 4 — Day 20): integrate with full output writer.
    """
    bins: dict[str, list[SeqRecord]] = {}
    for record, label in zip(records, labels, strict=True):
        bins.setdefault(label, []).append(record)
    return bins
