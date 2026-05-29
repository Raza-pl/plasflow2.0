"""ARG (Antibiotic Resistance Gene) detection via DIAMOND + CARD and/or SARG.

Pipeline (CARD-only):
    FASTA → call_orfs() → proteins.faa → run_diamond(card_db) → TSV
          → parse_diamond_hits() → [ARGHit(source="CARD")]

Pipeline (CARD + SARG dual):
    FASTA → call_orfs() → proteins.faa
          → run_diamond(card_db)  → card_hits.tsv  → parse_diamond_hits()
          → run_diamond(sarg_db)  → sarg_hits.tsv  → parse_sarg_hits()
          → merge_arg_hits()      → deduplicated [ARGHit] (CARD preferred per ORF)

Database setup (one-time):
    # CARD
    python -c "from plasflow2.annotate.args import setup_card_db; \\
               setup_card_db('data/databases/card')"

    # SARG — download from https://smile.hku.hk/SARGs, then:
    diamond makedb --in sarg.fasta -d data/databases/sarg/sarg
"""

from __future__ import annotations

import csv
import logging
import re
import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DIAMOND hit filters
# ---------------------------------------------------------------------------

# CARD: 80 % identity / 80 % coverage — the accepted standard for environmental
# metagenomics (RGI "Loose" mode, SARG, ResFinder metagenomic mode all use ~80 %).
# At 90 % (clinical-isolate standard) we miss the divergent ARG variants that
# dominate WWTP, soil, and aquatic metagenomes.
CARD_MIN_IDENTITY = 80.0
CARD_MIN_COVERAGE = 80.0

# SARG uses the same cutoffs — unified with CARD for consistency.
SARG_MIN_IDENTITY = 80.0
SARG_MIN_COVERAGE = 80.0

# ---------------------------------------------------------------------------
# Header regexes
# ---------------------------------------------------------------------------

# CARD sseqid format:  gb|PROT_ACC|ARO:XXXXX|GENE_NAME [Organism]
_CARD_SSEQID_RE = re.compile(r"gb\|(?P<prot_acc>[^|]+)\|(?P<aro>ARO:\d+)\|(?P<gene>[^\s\[]+)")

# SARG sseqid format:  SARG|drug_type|gene_family[*]|WP_accession
# e.g.  SARG|beta-lactam|bla*|WP_459377734.1
#       SARG|polymyxin|mcr*|WP_000001234.1
#       SARG|aminoglycoside|aph(6)*|WP_000005678.1
_SARG_SSEQID_RE = re.compile(
    r"SARG\|(?P<drug_type>[^|]+)\|(?P<gene_family>[^|*]+)\*?\|(?P<accession>\S+)"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ARGHit:
    """Single DIAMOND hit against CARD or SARG, annotated with resistance metadata."""

    contig_id: str
    gene_name: str  # e.g. "NDM-6", "TEM-1", "mcr-1"
    aro_accession: str  # ARO:XXXXXX for CARD hits; SARG acc for SARG hits
    amr_family: str  # e.g. "NDM beta-lactamase" / SARG subtype
    drug_class: str  # e.g. "carbapenem antibiotic; cephalosporin"
    resistance_mechanism: str  # e.g. "antibiotic inactivation"
    identity: float  # % amino-acid identity
    coverage: float  # % query coverage
    evalue: float
    source: Literal["CARD", "SARG"] = "CARD"
    # Internal: ORF id used for deduplication, not exposed in reports
    _orf_id: str = field(default="", repr=False, compare=False)


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
            [
                "diamond",
                "makedb",
                "--in",
                str(protein_fasta),
                "-d",
                str(card_dir / "card"),
            ],
            check=True,
        )
        logger.info("DIAMOND database written to %s", diamond_db)
    else:
        logger.info("DIAMOND database already exists: %s", diamond_db)

    return diamond_db, aro_index


# ---------------------------------------------------------------------------
# CARD metadata
# ---------------------------------------------------------------------------


def load_card_metadata(aro_index_path: Path | str) -> dict[str, dict]:  # type: ignore[type-arg]
    """Load CARD aro_index.tsv into a dict keyed by ARO accession.

    Args:
        aro_index_path: Path to aro_index.tsv (extracted from card.tar.bz2).

    Returns:
        Dict mapping "ARO:XXXXXX" → {gene, family, drug_class, mechanism}.
    """
    metadata: dict[str, dict] = {}  # type: ignore[type-arg]
    with open(aro_index_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            aro = row.get("ARO Accession", "").strip()
            if not aro:
                continue
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

    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    sequences = [str(r.seq) for r in records]
    contig_ids = [r.id for r in records]

    gene_finder = pyrodigal.GeneFinder(meta=True)

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
# DIAMOND search (shared by CARD and SARG)
# ---------------------------------------------------------------------------


def run_diamond(
    protein_fasta: Path | str,
    db: Path | str,
    out_tsv: Path | str,
    threads: int = 8,
    min_identity: float = CARD_MIN_IDENTITY,
    min_coverage: float = CARD_MIN_COVERAGE,
) -> Path:
    """Run DIAMOND BLASTp against a protein database.

    Args:
        protein_fasta: ORF-called protein sequences (from call_orfs).
        db: Path to DIAMOND-formatted database (.dmnd file).
        out_tsv: Output path for DIAMOND tabular results.
        threads: Number of CPU threads.
        min_identity: Minimum % amino-acid identity to report.
        min_coverage: Minimum % query coverage to report.

    Returns:
        Path to the TSV output file.
    """
    protein_fasta = Path(protein_fasta)
    db = Path(db)
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    db_stem = str(db).removesuffix(".dmnd")

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
        "1",
    ]
    logger.info("Running DIAMOND: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("DIAMOND stderr: %s", result.stderr)
        result.check_returncode()
    return out_tsv


# ---------------------------------------------------------------------------
# CARD hit parser
# ---------------------------------------------------------------------------


def parse_diamond_hits(
    tsv_path: Path | str,
    card_metadata: dict[str, dict] | None = None,  # type: ignore[type-arg]
) -> list[ARGHit]:
    """Parse DIAMOND tabular output from a CARD search into ARGHit objects.

    Args:
        tsv_path: Path to DIAMOND tabular output (format 6 with columns:
                  qseqid sseqid pident qcovhsp evalue stitle).
        card_metadata: Optional dict from load_card_metadata().

    Returns:
        List of ARGHit with source="CARD".
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
            qseqid, sseqid, pident, qcovhsp, evalue = (
                parts[0],
                parts[1],
                parts[2],
                parts[3],
                parts[4],
            )

            contig_id = re.sub(r"_\d+$", "", qseqid)

            match = _CARD_SSEQID_RE.search(sseqid)
            if match:
                aro = match.group("aro")
                gene_name = match.group("gene")
            else:
                aro = "unknown"
                gene_name = sseqid

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
                    source="CARD",
                    _orf_id=qseqid,
                )
            )

    logger.info("Parsed %d CARD ARG hits from %s", len(hits), tsv_path)
    return hits


# ---------------------------------------------------------------------------
# SARG hit parser
# ---------------------------------------------------------------------------


def parse_sarg_hits(tsv_path: Path | str) -> list[ARGHit]:
    """Parse DIAMOND tabular output from a SARG search into ARGHit objects.

    SARG FASTA headers use pipe-delimited fields:
        SARG|drug_type|gene_family[*]|WP_accession description
    e.g.
        SARG|beta-lactam|bla*|WP_459377734.1 MULTISPECIES: class A beta-lactamase
        SARG|polymyxin|mcr*|WP_000001234.1 mobile colistin resistance protein
        SARG|aminoglycoside|aph(6)*|WP_000005678.1 aminoglycoside phosphotransferase

    Field mapping:
        gene_family (stripped of trailing *) → gene_name  (e.g. "bla", "mcr", "aph(6)")
        drug_type                            → drug_class
        WP_accession                         → aro_accession
        gene_family                          → amr_family

    If the sseqid does not match this format, drug_class falls back to
    "unknown" and gene_name is taken from the last pipe-delimited token.

    Args:
        tsv_path: Path to DIAMOND tabular output (format 6):
                  qseqid sseqid pident qcovhsp evalue stitle.

    Returns:
        List of ARGHit with source="SARG".
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
            qseqid, sseqid, pident, qcovhsp, evalue, stitle = (
                parts[0],
                parts[1],
                parts[2],
                parts[3],
                parts[4],
                parts[5],
            )

            contig_id = re.sub(r"_\d+$", "", qseqid)

            # Try sseqid first, then fall back to stitle (which contains the full header)
            match = _SARG_SSEQID_RE.search(sseqid) or _SARG_SSEQID_RE.search(stitle)
            if match:
                drug_type = match.group("drug_type").strip()
                gene_family = match.group("gene_family").strip()
                accession = match.group("accession").strip()
                # gene_family is already stripped of trailing * by the regex
                gene_name = gene_family  # e.g. "bla", "mcr", "aph(6)"
                amr_family_val = gene_family
                aro_accession_val = accession  # e.g. "WP_459377734.1"
            else:
                drug_type = "unknown"
                gene_name = sseqid.split("|")[-1] if "|" in sseqid else sseqid
                amr_family_val = "unknown"
                aro_accession_val = sseqid.split("|")[-1] if "|" in sseqid else sseqid

            hits.append(
                ARGHit(
                    contig_id=contig_id,
                    gene_name=gene_name,
                    aro_accession=aro_accession_val,
                    amr_family=amr_family_val,
                    drug_class=drug_type,
                    resistance_mechanism="unknown",
                    identity=float(pident),
                    coverage=float(qcovhsp),
                    evalue=float(evalue),
                    source="SARG",
                    _orf_id=qseqid,
                )
            )

    logger.info("Parsed %d SARG ARG hits from %s", len(hits), tsv_path)
    return hits


# ---------------------------------------------------------------------------
# Dual-database merge
# ---------------------------------------------------------------------------


def merge_arg_hits(
    card_hits: list[ARGHit],
    sarg_hits: list[ARGHit],
) -> list[ARGHit]:
    """Merge CARD and SARG hit lists, preferring CARD when both detect the same ORF.

    Deduplication is per-ORF (_orf_id): if an ORF produced a CARD hit it is
    kept and the corresponding SARG hit for that ORF is discarded.  SARG hits
    for ORFs *not* found by CARD are appended as supplementary hits.

    Args:
        card_hits: Hits from parse_diamond_hits() (source="CARD").
        sarg_hits: Hits from parse_sarg_hits() (source="SARG").

    Returns:
        Merged list: all CARD hits + SARG-only hits, stable-ordered.
    """
    card_orf_ids: set[str] = {h._orf_id for h in card_hits if h._orf_id}
    sarg_only = [h for h in sarg_hits if h._orf_id not in card_orf_ids]

    merged = card_hits + sarg_only
    logger.info(
        "Merged ARG hits: %d CARD + %d SARG-only = %d total",
        len(card_hits),
        len(sarg_only),
        len(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# Convenience: full annotation for one FASTA
# ---------------------------------------------------------------------------


def annotate_contigs(
    fasta_path: Path | str,
    card_db: Path | str,
    aro_index_path: Path | str,
    work_dir: Path | str,
    threads: int = 8,
    sarg_db: Path | str | None = None,
    min_identity: float = CARD_MIN_IDENTITY,
    min_coverage: float = CARD_MIN_COVERAGE,
) -> list[ARGHit]:
    """End-to-end ARG annotation: ORF prediction → DIAMOND → parsed hits.

    When *sarg_db* is provided the function runs a second DIAMOND search
    against the SARG database and supplements CARD hits with any genes found
    only in SARG (see merge_arg_hits() for the deduplication logic).

    Args:
        fasta_path: Nucleotide FASTA of contigs to annotate.
        card_db: Path to DIAMOND .dmnd database built from CARD proteins.
        aro_index_path: Path to CARD aro_index.tsv.
        work_dir: Directory for intermediate files.
        threads: CPU threads for DIAMOND.
        sarg_db: Optional path to a DIAMOND .dmnd database built from SARG
                 (download SARG FASTA from https://smile.hku.hk/SARGs then
                  run: diamond makedb --in sarg.fasta -d sarg).
        min_identity: Minimum amino-acid identity % for DIAMOND hits (default
            80 %).  Use 90 % for clinical-isolate precision; 80 % is the
            standard for environmental/metagenomic samples.
        min_coverage: Minimum query coverage % for DIAMOND hits (default 80 %).

    Returns:
        List of ARGHit across all contigs.  Hits from CARD have source="CARD";
        SARG-only supplements have source="SARG".
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    proteins_path = work_dir / "proteins.faa"
    card_tsv = work_dir / "card_hits.tsv"

    call_orfs(fasta_path, proteins_path)
    run_diamond(
        proteins_path,
        card_db,
        card_tsv,
        threads=threads,
        min_identity=min_identity,
        min_coverage=min_coverage,
    )
    metadata = load_card_metadata(aro_index_path)
    card_hits = parse_diamond_hits(card_tsv, metadata)

    if sarg_db is not None:
        sarg_db_path = Path(sarg_db)
        if sarg_db_path.exists() or sarg_db_path.with_suffix(".dmnd").exists():
            sarg_tsv = work_dir / "sarg_hits.tsv"
            run_diamond(
                proteins_path,
                sarg_db_path,
                sarg_tsv,
                threads=threads,
                min_identity=min_identity,
                min_coverage=min_coverage,
            )
            sarg_hits = parse_sarg_hits(sarg_tsv)
            return merge_arg_hits(card_hits, sarg_hits)
        else:
            logger.warning("SARG database not found at %s — running CARD-only annotation", sarg_db)

    return card_hits
