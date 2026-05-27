"""Contig taxonomy annotation via DIAMOND + LCA (Kaiju-style).

Week 4 — Day 26 implementation.

Pipeline:
    FASTA → run_diamond_taxonomy() (blastx against GTDB/RefSeq protein DB)
          → parse_diamond_taxonomy_output() → [TaxHit per contig]
          → lca_for_contig() → TaxResult (lowest common ancestor)
          → assign_taxonomy() → dict[contig_id → TaxResult]

Database setup (one-time, example with GTDB-r220):
    # Download GTDB representative protein sequences
    wget https://data.ace.uq.edu.au/public/gtdb/data/releases/release220/220.0/genomic_files_reps/gtdb_proteins_aa_reps_r220.tar.gz
    tar xf gtdb_proteins_aa_reps_r220.tar.gz
    # Build DIAMOND database (headers include GTDB lineage)
    diamond makedb --in gtdb_prot_reps_r220.faa -d data/databases/gtdb/gtdb_r220.dmnd
    # Build taxon map from the GTDB metadata
    python -c "
    from plasflow2.annotate.taxonomy import build_gtdb_taxon_map
    build_gtdb_taxon_map('bac120_taxonomy_r220.tsv', 'data/databases/gtdb/taxon_map.tsv')
    "

Alternatively, use NCBI RefSeq with the prot.accession2taxid approach.

LCA algorithm (Kaiju-style):
    1. Collect top-N DIAMOND hits for a contig.
    2. Parse each hit's lineage into ranked levels:
       d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria; ...
    3. Walk from the deepest rank upwards until ≥ min_agreement fraction
       of hits agree at that rank.
    4. Return the deepest agreeing rank as the LCA.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# DIAMOND filters for taxonomy (more permissive than ARG detection)
TAX_MIN_IDENTITY = 70.0  # % amino-acid identity
TAX_MIN_COVERAGE = 60.0  # % query coverage
TAX_TOP_N = 10  # number of top hits per contig to use for LCA
TAX_MIN_AGREEMENT = 0.5  # fraction of hits that must agree at a rank

# GTDB rank prefixes in order from broadest to most specific
GTDB_RANK_PREFIXES = ["d__", "p__", "c__", "o__", "f__", "g__", "s__"]
GTDB_RANK_NAMES = ["domain", "phylum", "class", "order", "family", "genus", "species"]

# Regex: match any GTDB lineage embedded in a DIAMOND stitle
_LINEAGE_RE = re.compile(r"(d__[A-Za-z][^;]*(?:;[a-z]__[^;]*)*)")

# Map rank prefix → canonical rank name
_PREFIX_TO_RANK = dict(zip(GTDB_RANK_PREFIXES, GTDB_RANK_NAMES))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TaxHit:
    """Single DIAMOND hit carrying taxonomy lineage information."""

    contig_id: str
    accession: str  # subject accession from DIAMOND
    lineage: str  # raw lineage string, e.g. "d__Bacteria;p__...;s__..."
    identity: float  # % amino-acid identity
    coverage: float  # % query coverage
    evalue: float
    bit_score: float


@dataclass
class TaxResult:
    """Taxonomy assignment for one contig, determined by LCA."""

    contig_id: str
    lineage: str  # full LCA lineage, e.g. "d__Bacteria;p__Proteobacteria;..."
    rank: str  # lowest assigned rank name, e.g. "genus"
    taxon: str  # taxon name at that rank, e.g. "g__Klebsiella"
    num_hits: int = 0  # number of DIAMOND hits used to compute LCA
    agreement: float = 0.0  # fraction of hits agreeing at the LCA rank

    # Convenience properties
    @property
    def display(self) -> str:
        """Short human-readable label: rank + taxon, e.g. 'genus: g__Klebsiella'."""
        if not self.taxon:
            return "unclassified"
        return f"{self.rank}: {self.taxon}"

    @property
    def lineage_dict(self) -> dict[str, str]:
        """Return lineage as {rank_name: taxon_string} mapping."""
        out: dict[str, str] = {}
        for part in self.lineage.split(";"):
            part = part.strip()
            if not part:
                continue
            for prefix, rank_name in _PREFIX_TO_RANK.items():
                if part.startswith(prefix):
                    out[rank_name] = part
                    break
        return out


@dataclass
class TaxSummary:
    """Taxonomy summary across all contigs in a run."""

    total_contigs: int = 0
    classified: int = 0
    unclassified: int = 0
    # Count of contigs assigned at each rank
    rank_counts: dict[str, int] = field(default_factory=dict)
    # Count of contigs per top-level domain
    domain_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lineage parsing helpers
# ---------------------------------------------------------------------------


def parse_lineage(lineage_str: str) -> list[tuple[str, str]]:
    """Parse a GTDB-style lineage string into ordered (prefix, taxon) pairs.

    Args:
        lineage_str: e.g. "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;..."

    Returns:
        List of (prefix, taxon) in rank order, skipping empty/unknown levels.
        e.g. [("d__", "d__Bacteria"), ("p__", "p__Proteobacteria"), ...]
    """
    parts = [p.strip() for p in lineage_str.split(";") if p.strip()]
    result: list[tuple[str, str]] = []
    for part in parts:
        for prefix in GTDB_RANK_PREFIXES:
            if part.startswith(prefix):
                # Skip if taxon name is empty after prefix, or is a placeholder
                name = part[len(prefix) :]
                if name and name not in ("", "unknown", "unclassified", "?"):
                    result.append((prefix, part))
                break
    return result


def _extract_lineage_from_stitle(stitle: str) -> str:
    """Extract a GTDB lineage string embedded in a DIAMOND stitle field.

    GTDB protein FASTA headers look like:
      >ACC organism d__Bacteria;p__...;s__...
    DIAMOND puts the header description into stitle.

    Returns the lineage string, or "" if not found.
    """
    m = _LINEAGE_RE.search(stitle)
    if m:
        return m.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Taxon map (accession → lineage)
# ---------------------------------------------------------------------------


def build_gtdb_taxon_map(
    gtdb_taxonomy_tsv: Path | str,
    output_map: Path | str,
) -> Path:
    """Build a 2-column TSV (accession → lineage) from a GTDB taxonomy file.

    The GTDB genome_taxonomy file (e.g. bac120_taxonomy_r220.tsv) has format:
        GB_GCA_000001405.29  d__Bacteria;p__...;s__...

    This function converts it to a map suitable for lookup during LCA.
    Each representative's accession is mapped to its GTDB lineage.

    Args:
        gtdb_taxonomy_tsv: Path to GTDB bac120/ar53 taxonomy TSV.
        output_map: Output path for the 2-column accession→lineage map.

    Returns:
        Path to the written map file.
    """
    gtdb_taxonomy_tsv = Path(gtdb_taxonomy_tsv)
    output_map = Path(output_map)
    output_map.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(gtdb_taxonomy_tsv) as fin, open(output_map, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            acc = parts[0].strip()
            lineage = parts[1].strip()
            # Convert GTDB accession (GB_GCA_... or RS_GCF_...) to bare GCA/GCF
            acc_bare = re.sub(r"^(?:GB_|RS_)", "", acc)
            fout.write(f"{acc_bare}\t{lineage}\n")
            count += 1

    logger.info("Built taxon map with %d entries → %s", count, output_map)
    return output_map


def load_taxon_map(map_path: Path | str) -> dict[str, str]:
    """Load a 2-column accession→lineage TSV into a dict.

    Args:
        map_path: Path to taxon map (output of build_gtdb_taxon_map, or
                  any 2-column TSV with accession\tlineage).

    Returns:
        Dict mapping accession → lineage string.
    """
    taxon_map: dict[str, str] = {}
    with open(map_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                taxon_map[parts[0]] = parts[1]
    logger.info("Loaded %d entries from taxon map %s", len(taxon_map), map_path)
    return taxon_map


# ---------------------------------------------------------------------------
# DIAMOND search
# ---------------------------------------------------------------------------


def run_diamond_taxonomy(
    fasta_path: Path | str,
    taxonomy_db: Path | str,
    out_tsv: Path | str,
    threads: int = 8,
    mode: str = "blastx",
    min_identity: float = TAX_MIN_IDENTITY,
    min_coverage: float = TAX_MIN_COVERAGE,
    top_n: int = TAX_TOP_N,
    block_size: float = 0.5,
) -> Path:
    """Run DIAMOND blastx/blastp for taxonomy annotation.

    Args:
        fasta_path: Input nucleotide (blastx) or protein (blastp) FASTA.
        taxonomy_db: DIAMOND database (.dmnd) built from GTDB/RefSeq proteins.
        out_tsv: Output path for DIAMOND tabular results.
        threads: CPU threads for DIAMOND.
        mode: ``'blastx'`` (nucleotide input) or ``'blastp'`` (protein input).
        min_identity: Minimum % amino-acid identity to report.
        min_coverage: Minimum % query coverage to report.
        top_n: Maximum hits to return per query sequence (used for LCA).
        block_size: DIAMOND --block-size (lower = less RAM, default 0.5 ≈ 4 GB).

    Returns:
        Path to the TSV output file.

    Raises:
        subprocess.CalledProcessError: If DIAMOND fails.
        ValueError: If mode is not 'blastx' or 'blastp'.
    """
    if mode not in ("blastx", "blastp"):
        raise ValueError(f"mode must be 'blastx' or 'blastp', got: {mode!r}")

    fasta_path = Path(fasta_path)
    taxonomy_db = Path(taxonomy_db)
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    db_stem = str(taxonomy_db).removesuffix(".dmnd")

    cmd = [
        "diamond",
        mode,
        "--query",
        str(fasta_path),
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
        "bitscore",
        "stitle",
        "--id",
        str(min_identity),
        "--query-cover",
        str(min_coverage),
        "--threads",
        str(threads),
        "--max-target-seqs",
        str(top_n),
        "--block-size",
        str(block_size),
        "--sensitive",
    ]
    logger.info("Running DIAMOND taxonomy (%s): %s", mode, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("DIAMOND taxonomy stderr: %s", result.stderr)
        result.check_returncode()
    return out_tsv


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_diamond_taxonomy_output(
    tsv_path: Path | str,
    taxon_map: dict[str, str] | None = None,
    top_n: int = TAX_TOP_N,
) -> dict[str, list[TaxHit]]:
    """Parse DIAMOND tabular output into TaxHit objects grouped by contig.

    Lineage is resolved via (in priority order):
    1. The ``taxon_map`` dict (accession → lineage), if provided.
    2. Lineage embedded in the ``stitle`` field (GTDB FASTA headers contain it).
    3. Empty string if neither source has lineage info.

    Args:
        tsv_path: Path to DIAMOND output (format 6 columns:
                  qseqid sseqid pident qcovhsp evalue bitscore stitle).
        taxon_map: Optional dict from :func:`load_taxon_map`.
        top_n: Maximum hits to keep per contig (highest bitscore first).

    Returns:
        Dict mapping contig_id → list of TaxHit (sorted by bitscore desc).
    """
    tsv_path = Path(tsv_path)
    hits_by_contig: dict[str, list[TaxHit]] = {}

    with open(tsv_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            # Columns: qseqid sseqid pident qcovhsp evalue bitscore [stitle]
            qseqid = parts[0]
            sseqid = parts[1]
            pident = float(parts[2])
            qcovhsp = float(parts[3])
            evalue = float(parts[4])
            bitscore = float(parts[5])
            stitle = parts[6] if len(parts) > 6 else ""

            # Resolve contig ID (strip ORF suffix if running on translated ORFs)
            contig_id = re.sub(r"_\d+$", "", qseqid)

            # Resolve lineage
            lineage = ""
            if taxon_map:
                # Try full sseqid first, then stripped version
                lineage = taxon_map.get(sseqid, "")
                if not lineage:
                    acc_bare = re.sub(r"^(?:GB_|RS_|ref\||gb\|)", "", sseqid).split("|")[0]
                    lineage = taxon_map.get(acc_bare, "")
            if not lineage:
                lineage = _extract_lineage_from_stitle(stitle)

            hit = TaxHit(
                contig_id=contig_id,
                accession=sseqid,
                lineage=lineage,
                identity=pident,
                coverage=qcovhsp,
                evalue=evalue,
                bit_score=bitscore,
            )
            hits_by_contig.setdefault(contig_id, []).append(hit)

    # Sort by bitscore descending and keep top_n per contig
    for cid in hits_by_contig:
        hits_by_contig[cid].sort(key=lambda h: h.bit_score, reverse=True)
        hits_by_contig[cid] = hits_by_contig[cid][:top_n]

    total_hits = sum(len(v) for v in hits_by_contig.values())
    logger.info(
        "Parsed %d taxonomy hits for %d contigs from %s",
        total_hits,
        len(hits_by_contig),
        tsv_path,
    )
    return hits_by_contig


# ---------------------------------------------------------------------------
# LCA algorithm
# ---------------------------------------------------------------------------


def lca_for_contig(
    hits: list[TaxHit],
    min_agreement: float = TAX_MIN_AGREEMENT,
) -> TaxResult:
    """Compute the LCA taxonomy for a contig from its DIAMOND hits.

    The algorithm (Kaiju-style majority LCA):
    1. Parse each hit's lineage into ordered rank levels.
    2. Walk from the deepest rank (species) upwards.
    3. At each rank, count how many hits have a non-empty, consistent taxon.
    4. Return the deepest rank where ≥ min_agreement of all hits agree.

    Args:
        hits: List of TaxHit for one contig (already sorted by bitscore desc).
        min_agreement: Minimum fraction of hits that must agree at a rank
                       (default 0.5 = majority, Kaiju default).

    Returns:
        :class:`TaxResult` with the LCA lineage, rank, and taxon.
        If no agreement is found at any rank, returns an "unclassified" result.
    """
    if not hits:
        return TaxResult(
            contig_id="",
            lineage="",
            rank="unclassified",
            taxon="",
            num_hits=0,
            agreement=0.0,
        )

    contig_id = hits[0].contig_id
    n_hits = len(hits)

    # Parse lineages; skip hits with no lineage info
    parsed: list[list[tuple[str, str]]] = []
    for hit in hits:
        if hit.lineage:
            levels = parse_lineage(hit.lineage)
            if levels:
                parsed.append(levels)

    if not parsed:
        return TaxResult(
            contig_id=contig_id,
            lineage="",
            rank="unclassified",
            taxon="",
            num_hits=n_hits,
            agreement=0.0,
        )

    # Walk rank prefixes from broadest (d__) to most specific (s__)
    # At each level, count how many parsed lineages have a taxon at that level
    # and whether they agree.
    best_rank = "unclassified"
    best_taxon = ""
    best_lineage = ""
    best_agreement = 0.0

    for prefix, rank_name in zip(GTDB_RANK_PREFIXES, GTDB_RANK_NAMES):
        # Collect the taxon at this rank from each parsed hit
        taxa_at_rank: list[str] = []
        for levels in parsed:
            # Levels is a list of (prefix, taxon) in rank order
            taxon_at_level = ""
            for p, t in levels:
                if p == prefix:
                    taxon_at_level = t
                    break
            if taxon_at_level:
                taxa_at_rank.append(taxon_at_level)

        if not taxa_at_rank:
            continue  # no hits have this rank — skip

        # Find the most common taxon at this rank
        from collections import Counter

        taxon_counts = Counter(taxa_at_rank)
        most_common_taxon, most_common_count = taxon_counts.most_common(1)[0]

        # Agreement = fraction of ALL hits (not just those with this rank present)
        # that support the most common taxon (Kaiju uses total hits as denominator)
        agreement = most_common_count / n_hits

        if agreement > min_agreement:
            # Strict majority: this rank is assignable — continue to see if a deeper
            # rank also passes.  Using strict `>` (not `>=`) ensures that an exact
            # 50/50 tie stops at the parent rank rather than arbitrarily picking one
            # branch and continuing deeper (Kaiju default: ties are resolved upward).
            best_rank = rank_name
            best_taxon = most_common_taxon
            best_agreement = agreement
            # Build consensus lineage up to this rank from the first agreeing hit
            agreeing_lineages = [
                levels
                for levels in parsed
                if any(p == prefix and t == most_common_taxon for p, t in levels)
            ]
            if agreeing_lineages:
                repr_levels = agreeing_lineages[0]
                truncated = []
                for p, t in repr_levels:
                    truncated.append(t)
                    if p == prefix:
                        break
                best_lineage = ";".join(truncated)
        else:
            # Agreement drops to or below threshold — stop here (don't go deeper)
            break

    return TaxResult(
        contig_id=contig_id,
        lineage=best_lineage,
        rank=best_rank,
        taxon=best_taxon,
        num_hits=n_hits,
        agreement=best_agreement,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def assign_taxonomy(
    fasta_path: Path | str,
    taxonomy_db: Path | str,
    work_dir: Path | str,
    taxon_map_path: Path | str | None = None,
    threads: int = 8,
    mode: str = "blastx",
    min_identity: float = TAX_MIN_IDENTITY,
    min_coverage: float = TAX_MIN_COVERAGE,
    top_n: int = TAX_TOP_N,
    min_agreement: float = TAX_MIN_AGREEMENT,
    block_size: float = 0.5,
) -> dict[str, TaxResult]:
    """End-to-end taxonomy assignment: DIAMOND → parse → LCA per contig.

    Args:
        fasta_path: Input nucleotide FASTA (contigs).
        taxonomy_db: DIAMOND database (.dmnd) built from GTDB/RefSeq proteins.
        work_dir: Directory for intermediate files (diamond_taxonomy.tsv).
        taxon_map_path: Optional path to 2-column accession→lineage TSV.
                        If None, lineage is parsed from DIAMOND stitle.
        threads: CPU threads for DIAMOND.
        mode: ``'blastx'`` (nucleotide) or ``'blastp'`` (protein).
        min_identity: Minimum % identity for DIAMOND hits.
        min_coverage: Minimum % query coverage for DIAMOND hits.
        top_n: Number of top hits per contig to use for LCA.
        min_agreement: Fraction of hits that must agree at a rank (LCA parameter).
        block_size: DIAMOND block size (controls RAM usage).

    Returns:
        Dict mapping contig_id → :class:`TaxResult`.
        Contigs with no DIAMOND hits are absent from the dict.
    """
    fasta_path = Path(fasta_path)
    taxonomy_db = Path(taxonomy_db)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    diamond_tsv = work_dir / "diamond_taxonomy.tsv"

    # 1. Run DIAMOND
    run_diamond_taxonomy(
        fasta_path=fasta_path,
        taxonomy_db=taxonomy_db,
        out_tsv=diamond_tsv,
        threads=threads,
        mode=mode,
        min_identity=min_identity,
        min_coverage=min_coverage,
        top_n=top_n,
        block_size=block_size,
    )

    # 2. Load taxon map if provided
    taxon_map: dict[str, str] | None = None
    if taxon_map_path is not None:
        taxon_map = load_taxon_map(taxon_map_path)

    # 3. Parse hits
    hits_by_contig = parse_diamond_taxonomy_output(
        tsv_path=diamond_tsv,
        taxon_map=taxon_map,
        top_n=top_n,
    )

    # 4. LCA per contig
    results: dict[str, TaxResult] = {}
    for contig_id, hits in hits_by_contig.items():
        tax_result = lca_for_contig(hits, min_agreement=min_agreement)
        tax_result.contig_id = contig_id
        results[contig_id] = tax_result

    classified = sum(1 for r in results.values() if r.rank != "unclassified")
    logger.info(
        "Taxonomy: %d / %d contigs classified (%.1f%%)",
        classified,
        len(results),
        100 * classified / len(results) if results else 0,
    )
    return results


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------


def summarise_taxonomy(taxonomy: dict[str, TaxResult], total_contigs: int) -> TaxSummary:
    """Compute summary statistics from a taxonomy result dict.

    Args:
        taxonomy: Output of :func:`assign_taxonomy`.
        total_contigs: Total number of input contigs (including those with no hits).

    Returns:
        :class:`TaxSummary` with counts by rank and domain.
    """
    from collections import Counter

    rank_ctr: Counter[str] = Counter()
    domain_ctr: Counter[str] = Counter()

    classified = 0
    for result in taxonomy.values():
        if result.rank != "unclassified":
            classified += 1
            rank_ctr[result.rank] += 1
            # Extract domain from lineage
            for part in result.lineage.split(";"):
                part = part.strip()
                if part.startswith("d__"):
                    domain_ctr[part] += 1
                    break

    return TaxSummary(
        total_contigs=total_contigs,
        classified=classified,
        unclassified=total_contigs - classified,
        rank_counts=dict(rank_ctr),
        domain_counts=dict(domain_ctr),
    )
