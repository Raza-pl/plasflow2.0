"""ARG (Antibiotic Resistance Gene) detection via DIAMOND + CARD.

Week 3 — Days 15–16 implementation.

Pipeline:
    FASTA → call_orfs() → protein FASTA → run_diamond() → TSV
         → parse_diamond_hits() → [ARGHit]

CARD database setup (one-time):
    python -c "from plasflow2.annotate.args import setup_card_db; setup_card_db('data/databases/card')"
"""

from __future__ import annotations

import csv
import logging
import re
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# DIAMOND hit filters (per plan §2.1)
MIN_IDENTITY = 90.0  # % amino-acid identity
MIN_COVERAGE = 80.0  # % query coverage

# Regex to extract fields from CARD FASTA header / DIAMOND sseqid
# Header format: gb|PROT_ACC|ARO:XXXXX|GENE_NAME [Organism]
_CARD_SSEQID_RE = re.compile(r"gb\|(?P<prot_acc>[^|]+)\|(?P<aro>ARO:\d+)\|(?P<gene>[^\s\[]+)")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ARGHit:
    """Single DIAMOND hit against CARD, annotated with resistance metadata."""

    contig_id: str
    gene_name: str  # e.g. "NDM-6", "TEM-1"
    aro_accession: str  # e.g. "ARO:3002356"
    amr_family: str  # e.g. "NDM beta-lactamase"
    drug_class: str  # e.g. "carbapenem antibiotic; cephalosporin"
    resistance_mechanism: str  # e.g. "antibiotic inactivation"
    identity: float  # % amino-acid identity
    coverage: float  # % query coverage
    evalue: float


@dataclass
class ORF:
    """Predicted open reading frame from pyrodigal."""

    contig_id: str
    orf_id: str  # full pyrodigal ID: <contig>_<n>
    sequence: str  # amino-acid sequence


# ---------------------------------------------------------------------------
# CARD database setup
# ---------------------------------------------------------------------------


def setup_card_db(card_dir: Path | str, force: bool = False) -> tuple[Path, Path]:
    """Extract the CARD tar archive and build a DIAMOND protein database.

    Args:
        card_dir: Directory containing card.tar.bz2.
        force: Re-extract and re-build even if outputs already exist.

    Returns:
        Tuple of (diamond_db_path, aro_index_path).

    Raises:
        FileNotFoundError: If card.tar.bz2 is absent.
        subprocess.CalledProcessError: If DIAMOND makedb fails.
    """
    card_dir = Path(card_dir)
    tar_path = card_dir / "card.tar.bz2"
    if not tar_path.exists():
        raise FileNotFoundError(f"CARD archive not found: {tar_path}")

    protein_fasta = card_dir / "protein_fasta_protein_homolog_model.fasta"
    aro_index = card_dir / "aro_index.tsv"
    diamond_db = card_dir / "card.dmnd"

    # Extract if needed
    if force or not protein_fasta.exists() or not aro_index.exists():
        logger.info("Extracting CARD archive: %s", tar_path)
        with tarfile.open(tar_path, "r:bz2") as tf:
            members = [
                m
                for m in tf.getmembers()
                if m.name.lstrip("./")
                in {
                    "protein_fasta_protein_homolog_model.fasta",
                    "aro_index.tsv",
                    "card.json",
                }
            ]
            for m in members:
                m.name = Path(m.name).name  # strip leading ./
                tf.extract(m, path=card_dir)
        logger.info("Extracted %d CARD files to %s", len(members), card_dir)
    else:
        logger.info("CARD files already extracted — skipping")

    # Build DIAMOND database
    if force or not diamond_db.exists():
        logger.info("Building DIAMOND database from %s …", protein_fasta)
        subprocess.run(
            ["diamond", "makedb", "--in", str(protein_fasta), "-d", str(card_dir / "card")],
            check=True,
        )
        logger.info("DIAMOND database written to %s", diamond_db)
    else:
        logger.info("DIAMOND database already exists: %s", diamond_db)

    return diamond_db, aro_index


# ---------------------------------------------------------------------------
# CARD metadata
# ---------------------------------------------------------------------------


def load_card_metadata(aro_index_path: Path | str) -> dict[str, dict]:
    """Load CARD aro_index.tsv into a dict keyed by ARO accession.

    Args:
        aro_index_path: Path to aro_index.tsv (extracted from card.tar.bz2).

    Returns:
        Dict mapping "ARO:XXXXXX" → {gene, family, drug_class, mechanism}.
    """
    metadata: dict[str, dict] = {}
    with open(aro_index_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            aro = row.get("ARO Accession", "").strip()
            if not aro:
                continue
            # Drug class field may contain semicolon-separated values
            drug_classes = [d.strip() for d in row.get("Drug Class", "").split(";") if d.strip()]
            metadata[aro] = {
                "gene": row.get("ARO Name", "unknown").strip(),
                "family": row.get("AMR Gene Family", "unknown").strip(),
                "drug_class": "; ".join(drug_classes) if drug_classes else "unknown",
                "mechanism": row.get("Resistance Mechanism", "unknown").strip(),
            }
    logger.info("Loaded %d CARD ARO entries from %s", len(metadata), aro_index_path)
    return metadata


# ---------------------------------------------------------------------------
# ORF prediction
# ---------------------------------------------------------------------------


def call_orfs(
    fasta_path: Path | str,
    out_proteins: Path | str,
    min_gene_length: int = 90,
) -> list[ORF]:
    """Predict protein-coding ORFs using pyrodigal.

    Args:
        fasta_path: Input nucleotide FASTA (contigs).
        out_proteins: Path to write predicted proteins in FASTA format.
        min_gene_length: Minimum gene length in nucleotides (default 90 = 30 aa).

    Returns:
        List of ORF objects with contig_id, orf_id, and amino-acid sequence.
    """
    try:
        import pyrodigal  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pyrodigal is required for ORF prediction: pip install pyrodigal"
        ) from exc

    from Bio import SeqIO  # type: ignore[import]

    fasta_path = Path(fasta_path)
    out_proteins = Path(out_proteins)
    out_proteins.parent.mkdir(parents=True, exist_ok=True)

    # Load all contigs, train gene finder on them
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    sequences = [str(r.seq) for r in records]
    contig_ids = [r.id for r in records]

    gene_finder = pyrodigal.GeneFinder(meta=True)  # meta mode for metagenomes

    orfs: list[ORF] = []
    with open(out_proteins, "w") as fh:
        for contig_id, seq in zip(contig_ids, sequences, strict=True):
            try:
                genes = gene_finder.find_genes(seq)
            except Exception as exc:
                logger.warning("pyrodigal failed on %s: %s", contig_id, exc)
                continue
            for i, gene in enumerate(genes, start=1):
                aa = gene.translate()
                if len(aa) * 3 < min_gene_length:
                    continue
                orf_id = f"{contig_id}_{i}"
                fh.write(f">{orf_id}\n{aa}\n")
                orfs.append(ORF(contig_id=contig_id, orf_id=orf_id, sequence=aa))

    logger.info("Predicted %d ORFs from %d contigs → %s", len(orfs), len(records), out_proteins)
    return orfs


# ---------------------------------------------------------------------------
# DIAMOND search
# ---------------------------------------------------------------------------


def run_diamond(
    protein_fasta: Path | str,
    card_db: Path | str,
    out_tsv: Path | str,
    threads: int = 8,
    min_identity: float = MIN_IDENTITY,
    min_coverage: float = MIN_COVERAGE,
) -> Path:
    """Run DIAMOND BLASTp against the CARD protein database.

    Args:
        protein_fasta: ORF-called protein sequences (from call_orfs).
        card_db: Path to DIAMOND-formatted CARD database (.dmnd file).
        out_tsv: Output path for DIAMOND tabular results.
        threads: Number of CPU threads.
        min_identity: Minimum % amino-acid identity to report.
        min_coverage: Minimum % query coverage to report.

    Returns:
        Path to the TSV output file.
    """
    protein_fasta = Path(protein_fasta)
    card_db = Path(card_db)
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    # Strip .dmnd extension if provided — DIAMOND adds it automatically
    db_stem = str(card_db).removesuffix(".dmnd")

    cmd = [
        "diamond",
        "blastp",
        "--query",
        str(protein_fasta),
        "--db",
        db_stem,
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
        "1",  # top hit per ORF
    ]
    logger.info("Running DIAMOND: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("DIAMOND stderr: %s", result.stderr)
        result.check_returncode()
    return out_tsv


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_diamond_hits(
    tsv_path: Path | str,
    card_metadata: dict[str, dict] | None = None,
) -> list[ARGHit]:
    """Parse DIAMOND tabular output into ARGHit objects.

    Extracts gene name and ARO accession directly from the CARD FASTA header
    embedded in sseqid / stitle. If card_metadata is provided, it is used to
    look up drug class, AMR family, and resistance mechanism; otherwise these
    fields fall back to "unknown".

    Args:
        tsv_path: Path to DIAMOND tabular output (format 6 with columns:
                  qseqid sseqid pident qcovhsp evalue stitle).
        card_metadata: Optional dict from load_card_metadata().

    Returns:
        List of ARGHit, one per passing hit line.
    """
    tsv_path = Path(tsv_path)
    hits: list[ARGHit] = []

    with open(tsv_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            qseqid, sseqid, pident, qcovhsp, evalue, _ = (
                parts[0],
                parts[1],
                parts[2],
                parts[3],
                parts[4],
                parts[5],
            )

            # Derive contig ID by stripping Prodigal ORF suffix (_N)
            contig_id = re.sub(r"_\d+$", "", qseqid)

            # Parse CARD fields from sseqid:
            # format: gb|PROT_ACC|ARO:XXXXX|GENE_NAME
            match = _CARD_SSEQID_RE.search(sseqid)
            if match:
                aro = match.group("aro")
                gene_name = match.group("gene")
            else:
                # Fall back to extracting from stitle
                aro = "unknown"
                gene_name = sseqid

            # Look up rich metadata if available
            meta = (card_metadata or {}).get(aro, {})

            hits.append(
                ARGHit(
                    contig_id=contig_id,
                    gene_name=meta.get("gene", gene_name),
                    aro_accession=aro,
                    amr_family=meta.get("family", "unknown"),
                    drug_class=meta.get("drug_class", "unknown"),
                    resistance_mechanism=meta.get("mechanism", "unknown"),
                    identity=float(pident),
                    coverage=float(qcovhsp),
                    evalue=float(evalue),
                )
            )

    logger.info("Parsed %d ARG hits from %s", len(hits), tsv_path)
    return hits


# ---------------------------------------------------------------------------
# Convenience: full annotation for one FASTA
# ---------------------------------------------------------------------------


def annotate_contigs(
    fasta_path: Path | str,
    card_db: Path | str,
    aro_index_path: Path | str,
    work_dir: Path | str,
    threads: int = 8,
) -> list[ARGHit]:
    """End-to-end ARG annotation: ORF prediction → DIAMOND → parsed hits.

    Args:
        fasta_path: Nucleotide FASTA of contigs to annotate.
        card_db: Path to DIAMOND .dmnd database.
        aro_index_path: Path to aro_index.tsv.
        work_dir: Directory for intermediate files (proteins.faa, diamond.tsv).
        threads: CPU threads for DIAMOND.

    Returns:
        List of ARGHit across all contigs.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    proteins_path = work_dir / "proteins.faa"
    diamond_tsv = work_dir / "diamond_hits.tsv"

    call_orfs(fasta_path, proteins_path)
    run_diamond(proteins_path, card_db, diamond_tsv, threads=threads)
    metadata = load_card_metadata(aro_index_path)
    return parse_diamond_hits(diamond_tsv, metadata)
