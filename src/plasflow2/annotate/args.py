"""ARG (Antibiotic Resistance Gene) detection via DIAMOND + CARD.

Week 3 — Day 16 implementation target.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# DIAMOND hit filters (per plan)
MIN_IDENTITY = 90.0  # %
MIN_COVERAGE = 80.0  # %


@dataclass
class ARGHit:
    """Single DIAMOND hit against CARD."""

    contig_id: str
    gene_name: str
    amr_family: str
    drug_class: str
    identity: float  # %
    coverage: float  # % query coverage
    evalue: float


def run_diamond(
    protein_fasta: Path,
    card_db: Path,
    out_tsv: Path,
    threads: int = 8,
) -> Path:
    """Run DIAMOND BLASTp against the CARD protein database.

    Args:
        protein_fasta: ORF-called protein sequences (from pyrodigal).
        card_db: Path to DIAMOND-formatted CARD database.
        out_tsv: Output path for DIAMOND tabular results.
        threads: Number of CPU threads.

    Returns:
        Path to the TSV output file.

    TODO (Day 16):
        - Build CARD diamond DB with: diamond makedb --in card.faa -d card
        - Parse CARD metadata JSON for drug class / AMR family lookup.
        - Stream results without loading full output into memory.
    """
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "diamond",
        "blastp",
        "--query",
        str(protein_fasta),
        "--db",
        str(card_db),
        "--out",
        str(out_tsv),
        "--outfmt",
        "6",
        "qseqid",
        "sseqid",
        "pident",
        "qcovhsp",
        "evalue",
        "stitle",
        "--id",
        str(MIN_IDENTITY),
        "--query-cover",
        str(MIN_COVERAGE),
        "--threads",
        str(threads),
        "--sensitive",
    ]
    logger.info("Running DIAMOND: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_tsv


def parse_diamond_hits(tsv_path: Path, card_metadata: dict[str, dict]) -> list[ARGHit]:
    """Parse DIAMOND TSV output into ARGHit objects.

    Args:
        tsv_path: Path to DIAMOND tabular output.
        card_metadata: Dict mapping CARD accession → {gene, family, drug_class}.

    Returns:
        List of ARGHit, one per passing hit.

    TODO (Day 16): implement full CARD metadata parsing.
    """
    hits: list[ARGHit] = []
    with open(tsv_path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            qseqid, sseqid, pident, qcovhsp, evalue, stitle = parts[:6]
            contig_id = qseqid.rsplit("_", 1)[0]  # strip Prodigal ORF suffix
            meta = card_metadata.get(sseqid, {})
            hits.append(
                ARGHit(
                    contig_id=contig_id,
                    gene_name=meta.get("gene", sseqid),
                    amr_family=meta.get("family", "unknown"),
                    drug_class=meta.get("drug_class", "unknown"),
                    identity=float(pident),
                    coverage=float(qcovhsp),
                    evalue=float(evalue),
                )
            )
    logger.info("Parsed %d ARG hits from %s", len(hits), tsv_path)
    return hits
