"""Virulence factor annotation via DIAMOND + VFDB (set A, validated VFs only).

Pipeline:
    plasmid FASTA → call_orfs() → proteins.faa
                  → run_diamond(vfdb.dmnd) → vfdb_hits.tsv
                  → parse_vfdb_hits() → [VFHit]

Database setup (one-time, handled by scripts/setup_databases.sh):
    wget http://www.mgc.ac.cn/VFs/Down/VFDB_setA_pro.fas.gz
    gunzip VFDB_setA_pro.fas.gz
    diamond makedb --in VFDB_setA_pro.fas -d data/databases/vfdb/vfdb_setA

VFDB set A contains only experimentally validated virulence factors (core dataset).
Set B (broader, unvalidated) is available but generates more false positives.

Header format example:
    >VFG000068(gb|AAD42099) mgtC [mgtC (VF0091)] [Salmonella enterica]
     ├── VFG000068: VFDB gene ID
     ├── gb|AAD42099: GenBank accession
     ├── mgtC: gene name
     ├── VF0091: virulence factor group ID
     └── Salmonella enterica: source organism
"""

from __future__ import annotations

import csv
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from plasflow2.annotate.args import call_orfs

logger = logging.getLogger(__name__)

# DIAMOND thresholds — 60 % identity is the VFDB community standard for
# detecting divergent virulence factor homologues in environmental metagenomes.
VFDB_MIN_IDENTITY = 60.0
VFDB_MIN_COVERAGE = 80.0

# Header regex for VFDB set A protein FASTA
# >VFG000068(gb|AAD42099) mgtC [mgtC (VF0091)] [Salmonella enterica]
_VFDB_HEADER_RE = re.compile(
    r"^(?P<vfg_id>VFG\d+)"
    r"\((?:gb|ref)\|(?P<accession>[^|)]+)\)"
    r"\s+(?P<gene_name>\S+)"
    r"(?:\s+\[(?P<vf_group>[^\]]+)\])?"
    r"(?:\s+\[(?P<organism>[^\]]+)\])?"
)


@dataclass
class VFHit:
    """Single DIAMOND hit against the VFDB virulence factor database."""

    contig_id: str
    gene_name: str  # e.g. "mgtC", "stx1A"
    vfg_id: str  # VFDB gene ID, e.g. "VFG000068"
    vf_group: str  # VF group name, e.g. "mgtC (VF0091)"
    organism: str  # Source organism, e.g. "Salmonella enterica"
    identity: float  # % amino-acid identity
    coverage: float  # % query coverage
    evalue: float


def _parse_vfdb_stitle(stitle: str) -> tuple[str, str, str, str]:
    """Parse gene_name, vfg_id, vf_group, organism from a VFDB stitle field.

    The stitle field in DIAMOND outfmt 6 is the full FASTA header minus the '>'.

    Returns:
        (gene_name, vfg_id, vf_group, organism) — any field may be empty string
        if the header doesn't match the expected pattern.
    """
    m = _VFDB_HEADER_RE.match(stitle.strip())
    if not m:
        # Fallback: use whatever is in the title as gene_name
        return stitle.strip()[:40], "", "", ""
    return (
        m.group("gene_name") or "",
        m.group("vfg_id") or "",
        m.group("vf_group") or "",
        m.group("organism") or "",
    )


def run_vfdb_diamond(
    protein_fasta: Path | str,
    vfdb: Path | str,
    out_tsv: Path | str,
    threads: int = 8,
    min_identity: float = VFDB_MIN_IDENTITY,
    min_coverage: float = VFDB_MIN_COVERAGE,
) -> None:
    """Run DIAMOND BLASTp against the VFDB database.

    Args:
        protein_fasta: ORF-called protein sequences (.faa).
        vfdb: Path to DIAMOND-formatted VFDB database (.dmnd).
        out_tsv: Output path for DIAMOND tabular results.
        threads: CPU threads.
        min_identity: Minimum amino-acid identity % (default 60).
        min_coverage: Minimum query coverage % (default 80).
    """
    protein_fasta = Path(protein_fasta)
    vfdb = Path(vfdb)
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "diamond",
        "blastp",
        "--query",
        str(protein_fasta),
        "--db",
        str(vfdb),
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
        str(min_identity),
        "--query-cover",
        str(min_coverage),
        "--threads",
        str(threads),
        "--sensitive",
        "--max-target-seqs",
        "1",
    ]
    logger.info("Running DIAMOND (VFDB): %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("DIAMOND (VFDB) failed: %s", result.stderr[:500])
        raise RuntimeError(f"DIAMOND (VFDB) failed with exit code {result.returncode}")


def parse_vfdb_hits(tsv_path: Path | str) -> list[VFHit]:
    """Parse DIAMOND tabular output against VFDB into VFHit objects.

    Args:
        tsv_path: Path to DIAMOND outfmt 6 TSV (qseqid sseqid pident qcovhsp evalue stitle).

    Returns:
        List of VFHit, one per row.
    """
    tsv_path = Path(tsv_path)
    hits: list[VFHit] = []

    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        return hits

    with open(tsv_path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 6:
                continue
            qseqid, _sseqid, pident, qcovhsp, evalue, stitle = row[:6]
            # Contig ID is the ORF id minus the trailing _<n> suffix
            contig_id = "_".join(qseqid.rsplit("_", 1)[:-1]) if "_" in qseqid else qseqid
            gene_name, vfg_id, vf_group, organism = _parse_vfdb_stitle(stitle)
            hits.append(
                VFHit(
                    contig_id=contig_id,
                    gene_name=gene_name,
                    vfg_id=vfg_id,
                    vf_group=vf_group,
                    organism=organism,
                    identity=float(pident),
                    coverage=float(qcovhsp),
                    evalue=float(evalue),
                )
            )

    logger.info("Parsed %d VFDB virulence factor hits from %s", len(hits), tsv_path)
    return hits


def annotate_vf(
    fasta_path: Path | str,
    vfdb: Path | str,
    work_dir: Path | str,
    threads: int = 8,
    min_identity: float = VFDB_MIN_IDENTITY,
    min_coverage: float = VFDB_MIN_COVERAGE,
    reuse_proteins: Path | str | None = None,
) -> list[VFHit]:
    """End-to-end virulence factor annotation: ORF prediction → DIAMOND → hits.

    Args:
        fasta_path: Nucleotide FASTA of plasmid contigs.
        vfdb: Path to DIAMOND .dmnd database built from VFDB set A proteins.
        work_dir: Directory for intermediate files.
        threads: CPU threads for DIAMOND.
        min_identity: Minimum amino-acid identity % (default 60).
        min_coverage: Minimum query coverage % (default 80).
        reuse_proteins: If provided, skip ORF prediction and use this .faa file
            directly (e.g. already predicted for ARG annotation).

    Returns:
        List of VFHit across all contigs.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    proteins_path = Path(reuse_proteins) if reuse_proteins else work_dir / "proteins.faa"
    vfdb_tsv = work_dir / "vfdb_hits.tsv"

    if reuse_proteins is None:
        call_orfs(fasta_path, proteins_path)
    else:
        logger.info("Reusing pre-predicted ORFs from %s", proteins_path)

    run_vfdb_diamond(
        proteins_path,
        vfdb,
        vfdb_tsv,
        threads=threads,
        min_identity=min_identity,
        min_coverage=min_coverage,
    )
    return parse_vfdb_hits(vfdb_tsv)
