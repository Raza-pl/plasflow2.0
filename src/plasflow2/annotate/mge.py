"""Mobile Genetic Element (MGE) detection via DIAMOND + MGE protein database.

Detects:
  - Insertion sequences (IS elements) — single-module transposons with one or
    two transposase ORFs; the most common MGEs on plasmids.
  - Composite transposons — flanked by IS elements carrying cargo genes.
  - Complex transposons — e.g. Tn3 family with resolvase + transposase.
  - Integrons — intI1/intI2 integrase genes; important AMR gene mobilisers.

Pipeline:
    plasmid FASTA → call_orfs() → proteins.faa
                  → run_diamond(isfinder.dmnd) → mge_hits.tsv
                  → parse_mge_hits() → [MGEHit]

Database setup (one-time, handled by scripts/setup_databases.sh):
    # Pärnänen et al. 2018 MGE database — IS*, integrons, transposons from NCBI
    # CDS translated to protein, then DIAMOND database built:
    diamond makedb --in data/databases/mge/mge_proteins.faa \\
                   -d data/databases/mge/isfinder

Database: Pärnänen et al. (2018) Nature Communications 9:3891
    https://github.com/KatariinaParnanen/MobileGeneticElementDatabase
    - ~2,000 unique MGE CDS sequences (99% identity clustered)
    - Covers IS*, ISCR*, intI1/intI2 (integrons), tniA/B, tnpA (transposons),
      qacEdelta (quaternary ammonium resistance cassettes), Tn916-family ORFs
    - Sourced from NCBI nucleotide database annotations

Header format (NCBI-style gene name + accession):
    >IS1_1 gb|AAA62386.1| IS1 transposase [Escherichia coli]
    >intI1_1 gb|AAB59737.1| integron integrase IntI1 [E. coli]
    General pattern: >{gene_name}_{n} {description}
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

# 70 % identity / 80 % coverage — transposases diverge faster than housekeeping
# genes but the DDE catalytic domain is well conserved. 70 % captures divergent
# IS copies on environmental plasmids while limiting spurious hits to non-MGE
# DDE-fold proteins (e.g. RNase H, integrases).
MGE_MIN_IDENTITY = 70.0
MGE_MIN_COVERAGE = 80.0

# IS family inference — covers ISfinder names AND Pärnänen NCBI gene names
_IS_FAMILY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Specific IS families (check before generic IS prefix)
    (re.compile(r"\bIS26\b", re.I), "IS26"),
    (re.compile(r"\bIS21\b", re.I), "IS21"),
    (re.compile(r"\bIS30\b", re.I), "IS30"),
    (re.compile(r"\bIS66\b", re.I), "IS66"),
    (re.compile(r"\bIS91\b", re.I), "IS91"),
    (re.compile(r"\bIS110\b", re.I), "IS110"),
    (re.compile(r"\bIS200\b|\bIS605\b", re.I), "IS200/IS605"),
    (re.compile(r"\bIS256\b", re.I), "IS256"),
    (re.compile(r"\bIS630\b", re.I), "IS630"),
    (re.compile(r"\bISCR\b", re.I), "ISCR"),
    (re.compile(r"\bIS1\b", re.I), "IS1"),
    (re.compile(r"\bIS3\b", re.I), "IS3"),
    (re.compile(r"\bIS4\b", re.I), "IS4"),
    (re.compile(r"\bIS5\b", re.I), "IS5"),
    (re.compile(r"\bIS6\b", re.I), "IS6"),
    # Transposons
    (re.compile(r"\bTn3\b|\bTn903\b|\bTn1000\b", re.I), "Tn3"),
    (re.compile(r"\bTn10\b|\bTn5\b|\bTn7\b|\bTn916\b", re.I), "Complex Tn"),
    (re.compile(r"\btniA\b|\btniB\b|\btnpA\b|\btnpR\b", re.I), "Transposon"),
    # Integrons (intI1/intI2 integrase, istA/istB cassette genes)
    (re.compile(r"\bintegron\b|\bintI\b|\bintI1\b|\bintI2\b", re.I), "Integron"),
    (re.compile(r"\bistA\b|\bistB\b", re.I), "Integron"),
    # Other MGE types
    (re.compile(r"\bqacE\b|\bqacEdelta\b", re.I), "qacE/Integron"),
    (re.compile(r"\bMITE\b", re.I), "MITE"),
]


def _infer_is_family(name: str, description: str) -> str:
    """Infer the IS/MGE family from element name and description.

    Handles both ISfinder-style names (ISAba1, IS26) and Pärnänen database
    NCBI-style gene names (intI1_1, tniA_5, IS1_3).
    """
    # Strip trailing _<number> suffix common in Pärnänen headers (e.g. "IS1_3" → "IS1")
    clean_name = re.sub(r"_\d+$", "", name)
    text = f"{clean_name} {description}"
    for pattern, family in _IS_FAMILY_PATTERNS:
        if pattern.search(text):
            return family
    # Generic IS prefix fallback (e.g. ISAba1 → "ISAba", ISSoc5 → "ISSoc")
    m = re.match(r"^IS([A-Za-z0-9]{1,4})", clean_name, re.I)
    if m:
        return f"IS{m.group(1)}"
    return "Unknown"


@dataclass
class MGEHit:
    """Single DIAMOND hit against the ISfinder MGE protein database."""

    contig_id: str
    is_name: str  # ISfinder element name, e.g. "ISAba1"
    is_family: str  # IS family, e.g. "IS4", "Tn3", "Integron"
    description: str  # Free-text description from ISfinder header
    identity: float  # % amino-acid identity to ISfinder reference
    coverage: float  # % query coverage
    evalue: float


def _parse_isfinder_stitle(stitle: str) -> tuple[str, str]:
    """Extract IS element name and description from ISfinder DIAMOND stitle.

    ISfinder headers are free-form but typically start with the IS name:
        ISAba1 AcinetobacterBA...  → ("ISAba1", "Acinetobacter ...")
        IS26 transposase IS26      → ("IS26", "transposase IS26")
        ISSoc5 IS5 family ...      → ("ISSoc5", "IS5 family ...")

    Returns:
        (is_name, description)
    """
    stitle = stitle.strip()
    # IS name is the first whitespace-delimited token if it matches IS/Tn/MITE pattern
    parts = stitle.split(None, 1)
    if not parts:
        return stitle, ""
    first = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    if re.match(r"^(IS|Tn|MITE|ICE|IME|CRISPRas)", first, re.I):
        return first, rest
    return first, rest


def run_mge_diamond(
    protein_fasta: Path | str,
    mge_db: Path | str,
    out_tsv: Path | str,
    threads: int = 8,
    min_identity: float = MGE_MIN_IDENTITY,
    min_coverage: float = MGE_MIN_COVERAGE,
) -> None:
    """Run DIAMOND BLASTp against the ISfinder/MGE protein database."""
    protein_fasta = Path(protein_fasta)
    mge_db = Path(mge_db)
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "diamond",
        "blastp",
        "--query",
        str(protein_fasta),
        "--db",
        str(mge_db),
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
    logger.info("Running DIAMOND (ISfinder/MGE): %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("DIAMOND (MGE) failed: %s", result.stderr[:500])
        raise RuntimeError(f"DIAMOND (MGE) failed with exit code {result.returncode}")


def parse_mge_hits(tsv_path: Path | str) -> list[MGEHit]:
    """Parse DIAMOND output against ISfinder into MGEHit objects."""
    tsv_path = Path(tsv_path)
    hits: list[MGEHit] = []

    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        return hits

    with open(tsv_path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 6:
                continue
            qseqid, _sseqid, pident, qcovhsp, evalue, stitle = row[:6]
            contig_id = "_".join(qseqid.rsplit("_", 1)[:-1]) if "_" in qseqid else qseqid
            is_name, description = _parse_isfinder_stitle(stitle)
            is_family = _infer_is_family(is_name, description)
            hits.append(
                MGEHit(
                    contig_id=contig_id,
                    is_name=is_name,
                    is_family=is_family,
                    description=description[:120],
                    identity=float(pident),
                    coverage=float(qcovhsp),
                    evalue=float(evalue),
                )
            )

    logger.info("Parsed %d MGE hits from %s", len(hits), tsv_path)
    return hits


def annotate_mge(
    fasta_path: Path | str,
    mge_db: Path | str,
    work_dir: Path | str,
    threads: int = 8,
    min_identity: float = MGE_MIN_IDENTITY,
    min_coverage: float = MGE_MIN_COVERAGE,
    reuse_proteins: Path | str | None = None,
) -> list[MGEHit]:
    """End-to-end MGE annotation: ORF prediction → DIAMOND → parsed hits.

    Args:
        fasta_path: Nucleotide FASTA of plasmid contigs.
        mge_db: Path to DIAMOND .dmnd database built from ISfinder proteins.
        work_dir: Directory for intermediate files.
        threads: CPU threads for DIAMOND.
        min_identity: Minimum amino-acid identity % (default 70).
        min_coverage: Minimum query coverage % (default 80).
        reuse_proteins: Reuse pre-predicted ORF .faa (from ARG annotation step)
            to avoid running pyrodigal twice.

    Returns:
        List of MGEHit across all contigs.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    proteins_path = Path(reuse_proteins) if reuse_proteins else work_dir / "proteins.faa"
    mge_tsv = work_dir / "mge_hits.tsv"

    if reuse_proteins is None:
        call_orfs(fasta_path, proteins_path)
    else:
        logger.info("Reusing pre-predicted ORFs from %s", proteins_path)

    run_mge_diamond(
        proteins_path,
        mge_db,
        mge_tsv,
        threads=threads,
        min_identity=min_identity,
        min_coverage=min_coverage,
    )
    return parse_mge_hits(mge_tsv)
