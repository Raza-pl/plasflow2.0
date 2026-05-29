"""End-to-end PlasFlow v2 pipeline.

Week 3 — Day 18 implementation.

Orchestrates:
    FASTA → classify (MLP) → [plasmid contigs]
                           → annotate ARGs (DIAMOND/CARD)
                           → annotate mobility (MOB-suite)
                           → risk score (scorer.py)
                           → PipelineResult

Typical usage:
    from plasflow2.pipeline import run_pipeline
    result = run_pipeline(
        fasta_path="contigs.fasta",
        model_path="data/models/mlp_v2.pt",
        card_db="data/databases/card/card.dmnd",
        aro_index="data/databases/card/aro_index.tsv",
        work_dir="output/run1",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from Bio.SeqRecord import SeqRecord  # type: ignore[import]

from plasflow2.annotate.args import ARGHit, annotate_contigs
from plasflow2.annotate.mge import MGEHit, annotate_mge
from plasflow2.annotate.mobility import (
    MobilityResult,
    index_by_contig,
    parse_mob_results,
    run_mob_typer,
)
from plasflow2.annotate.taxonomy import TaxResult, assign_taxonomy
from plasflow2.annotate.vfdb import VFHit, annotate_vf
from plasflow2.classify.predict import Prediction, predict
from plasflow2.risk.scorer import RiskScore, score_plasmid
from plasflow2.utils.fasta import load_fasta, write_fasta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ContigResult:
    """All annotations for a single contig that passed the plasmid filter."""

    record: SeqRecord
    prediction: Prediction
    arg_hits: list[ARGHit]
    mobility: MobilityResult | None
    risk: RiskScore
    taxonomy: TaxResult | None = None  # LCA taxonomy from DIAMOND (optional)
    vf_hits: list[VFHit] = field(default_factory=list)  # VFDB virulence factors
    mge_hits: list[MGEHit] = field(default_factory=list)  # ISfinder MGE elements


@dataclass
class NonPlasmidContigResult:
    """Prediction + taxonomy for a chromosome / phage / archaea / unclassified contig.

    Does not carry ARG, mobility, or risk annotations (plasmid-only steps).
    The taxonomy field is populated when --skip-taxonomy is False and a
    DIAMOND taxonomy database is available — same LCA pipeline as plasmids.
    """

    record: SeqRecord
    prediction: Prediction
    taxonomy: TaxResult | None = None


@dataclass
class PipelineResult:
    """Aggregated results for one run_pipeline() call."""

    input_fasta: Path
    all_predictions: list[Prediction]  # every contig, all classes
    plasmid_results: list[ContigResult]  # plasmid contigs only, fully annotated
    # Non-plasmid contigs: chromosome, phage, archaea, unclassified — prediction + taxonomy only
    non_plasmid_results: list[NonPlasmidContigResult] = field(default_factory=list)
    # Taxonomy results for ALL contigs (keyed by contig_id); empty if skipped
    taxonomy: dict[str, TaxResult] = field(default_factory=dict)
    # Convenience counts
    class_counts: dict[str, int] = field(default_factory=dict)
    total_sequences: int = 0
    total_plasmids: int = 0
    total_args: int = 0

    def __post_init__(self) -> None:
        self.total_sequences = len(self.all_predictions)
        self.total_plasmids = len(self.plasmid_results)
        self.total_args = sum(len(cr.arg_hits) for cr in self.plasmid_results)

        if not self.class_counts:
            counts: dict[str, int] = {}
            for p in self.all_predictions:
                counts[p.label] = counts.get(p.label, 0) + 1
            self.class_counts = counts


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    fasta_path: Path | str,
    model_path: Path | str,
    card_db: Path | str,
    aro_index: Path | str,
    work_dir: Path | str,
    source_context: str = "unspecified",
    confidence_threshold: float = 0.70,
    plasmid_threshold: float = 0.95,
    min_contig_length: int = 1000,
    threads: int = 8,
    skip_mobility: bool = False,
    taxonomy_db: Path | str | None = None,
    taxon_map_path: Path | str | None = None,
    skip_taxonomy: bool = False,
    sarg_db: Path | str | None = None,
    min_identity: float = 80.0,
    vfdb: Path | str | None = None,
    mge_db: Path | str | None = None,
) -> PipelineResult:
    """Run the full PlasFlow v2 pipeline on a FASTA file.

    Steps
    -----
    1. Load and length-filter contigs from *fasta_path*.
    2. Classify every contig with the MLP (``predict()``).
    3. Write plasmid-classified contigs to ``work_dir/plasmids.fasta``.
    4. Annotate ARGs on plasmid contigs via DIAMOND + CARD.
    5. Annotate mobility on plasmid contigs via MOB-suite (unless
       *skip_mobility* is True — useful when mob_typer is unavailable).
    6. Score each plasmid contig with ``score_plasmid()``.

    Args:
        fasta_path: Input nucleotide FASTA (assembled contigs).
        model_path: Path to trained MLP weights (.pt file).
        card_db: Path to DIAMOND-formatted CARD database (.dmnd).
        aro_index: Path to CARD aro_index.tsv.
        work_dir: Directory for all intermediate and output files.
        source_context: Sample provenance for risk scoring — one of
            ``clinical``, ``wastewater``, ``environmental``,
            ``unspecified``.
        confidence_threshold: Minimum MLP confidence for chromosome / phage /
            archaea calls (sequences below this are labelled ``unclassified``).
        plasmid_threshold: Minimum MLP confidence for plasmid calls.  Defaults
            to 0.95 — higher than *confidence_threshold* — to compensate for
            class-prior imbalance: the model trains on ~25 % plasmid but real
            metagenomes contain only ~2–5 % plasmid.
        min_contig_length: Discard sequences shorter than this (bp).
        threads: CPU threads for DIAMOND and MOB-suite.
        skip_mobility: If True, skip mob_typer and set mobility to None
            for all contigs (use when mob_typer is not installed).
        taxonomy_db: Path to a DIAMOND database (.dmnd) built from GTDB-r220
            or RefSeq protein sequences for taxonomy annotation.  If ``None``
            and *skip_taxonomy* is False, taxonomy is skipped with a warning.
        taxon_map_path: Optional path to a 2-column accession→lineage TSV
            (output of ``build_gtdb_taxon_map``).  When None, lineage is
            extracted from DIAMOND ``stitle`` fields.
        skip_taxonomy: If True, skip taxonomy annotation entirely (useful when
            no GTDB/RefSeq database is available).
        sarg_db: Optional path to a DIAMOND .dmnd database built from the SARG
            (Structured ARG) database.  When provided, ARG annotation runs
            against both CARD and SARG; CARD hits are preferred per ORF and
            SARG contributes supplementary hits for genes not in CARD.
        min_identity: Minimum amino-acid identity % for DIAMOND ARG hits
            (default 80 %).  80 % is the standard for environmental/metagenomic
            samples; use 90 % for clinical-isolate precision.
        vfdb: Optional path to a DIAMOND .dmnd database built from VFDB set A
            protein sequences.  When provided, virulence factor annotation runs
            on plasmid contigs using the pre-predicted ORFs (no extra Prodigal
            pass needed).
        mge_db: Optional path to a DIAMOND .dmnd database built from ISfinder
            transposase protein sequences.  When provided, MGE/IS element
            annotation runs on plasmid contigs.

    Returns:
        :class:`PipelineResult` with all predictions and per-plasmid
        annotations.

    Raises:
        FileNotFoundError: If *fasta_path*, *model_path*, *card_db*, or
            *aro_index* do not exist.
    """
    fasta_path = Path(fasta_path)
    model_path = Path(model_path)
    card_db = Path(card_db)
    aro_index = Path(aro_index)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    taxonomy_db_path = Path(taxonomy_db) if taxonomy_db else None
    taxon_map = Path(taxon_map_path) if taxon_map_path else None

    for p, name in [
        (fasta_path, "fasta_path"),
        (model_path, "model_path"),
        (card_db, "card_db"),
        (aro_index, "aro_index"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    # ------------------------------------------------------------------
    # 1. Load contigs
    # ------------------------------------------------------------------
    logger.info("Loading contigs from %s (min_length=%d)", fasta_path, min_contig_length)
    records = load_fasta(fasta_path, min_length=min_contig_length)
    if not records:
        logger.warning("No sequences pass min_length=%d filter — aborting.", min_contig_length)
        return PipelineResult(
            input_fasta=fasta_path,
            all_predictions=[],
            plasmid_results=[],
        )

    sequences = [str(r.seq) for r in records]
    seq_ids = [r.id for r in records]

    # ------------------------------------------------------------------
    # 2. Classify
    # ------------------------------------------------------------------
    logger.info("Classifying %d contigs …", len(sequences))
    predictions = predict(
        sequences,
        seq_ids,
        model_path,
        threshold=confidence_threshold,
        plasmid_threshold=plasmid_threshold,
    )
    pred_by_id = {p.sequence_id: p for p in predictions}

    # ------------------------------------------------------------------
    # 3. Extract plasmid contigs
    # ------------------------------------------------------------------
    plasmid_records = [r for r in records if pred_by_id[r.id].label == "plasmid"]
    logger.info("Plasmid contigs: %d / %d", len(plasmid_records), len(records))

    if not plasmid_records:
        return PipelineResult(
            input_fasta=fasta_path,
            all_predictions=predictions,
            plasmid_results=[],
        )

    plasmid_fasta = work_dir / "plasmids.fasta"
    write_fasta(plasmid_records, plasmid_fasta)

    # ------------------------------------------------------------------
    # 4. ARG annotation
    # ------------------------------------------------------------------
    logger.info(
        "Annotating ARGs on %d plasmid contigs (CARD%s) …",
        len(plasmid_records),
        " + SARG" if sarg_db else "",
    )
    arg_hits = annotate_contigs(
        fasta_path=plasmid_fasta,
        card_db=card_db,
        aro_index_path=aro_index,
        work_dir=work_dir / "arg_annotation",
        threads=threads,
        sarg_db=sarg_db,
        min_identity=min_identity,
    )
    # Group hits by contig_id for fast lookup
    args_by_contig: dict[str, list[ARGHit]] = {}
    for hit in arg_hits:
        args_by_contig.setdefault(hit.contig_id, []).append(hit)

    # Pre-predicted proteins path — reused by VFDB and MGE to avoid running
    # pyrodigal three times on the same sequences.
    arg_proteins = work_dir / "arg_annotation" / "proteins.faa"

    # ------------------------------------------------------------------
    # 4b. Virulence factor annotation (VFDB set A)
    # ------------------------------------------------------------------
    vf_by_contig: dict[str, list[VFHit]] = {}
    if vfdb is not None:
        vfdb_path = Path(vfdb)
        if vfdb_path.exists() or vfdb_path.with_suffix(".dmnd").exists():
            logger.info(
                "Annotating virulence factors on %d plasmid contigs (VFDB) …", len(plasmid_records)
            )
            try:
                vf_hits = annotate_vf(
                    fasta_path=plasmid_fasta,
                    vfdb=vfdb_path,
                    work_dir=work_dir / "vfdb_annotation",
                    threads=threads,
                    reuse_proteins=arg_proteins if arg_proteins.exists() else None,
                )
                for hit in vf_hits:
                    vf_by_contig.setdefault(hit.contig_id, []).append(hit)
                logger.info(
                    "Found %d virulence factor hits across %d contigs",
                    len(vf_hits),
                    len(vf_by_contig),
                )
            except Exception as exc:
                logger.warning("VFDB annotation failed: %s — skipping.", exc)
        else:
            logger.warning("VFDB database not found at %s — skipping VF annotation.", vfdb)

    # ------------------------------------------------------------------
    # 4c. MGE / IS element annotation (ISfinder)
    # ------------------------------------------------------------------
    mge_by_contig: dict[str, list[MGEHit]] = {}
    if mge_db is not None:
        mge_db_path = Path(mge_db)
        if mge_db_path.exists() or mge_db_path.with_suffix(".dmnd").exists():
            logger.info("Annotating MGEs on %d plasmid contigs (ISfinder) …", len(plasmid_records))
            try:
                mge_hits = annotate_mge(
                    fasta_path=plasmid_fasta,
                    mge_db=mge_db_path,
                    work_dir=work_dir / "mge_annotation",
                    threads=threads,
                    reuse_proteins=arg_proteins if arg_proteins.exists() else None,
                )
                for hit in mge_hits:
                    mge_by_contig.setdefault(hit.contig_id, []).append(hit)
                logger.info(
                    "Found %d MGE hits across %d contigs", len(mge_hits), len(mge_by_contig)
                )
            except Exception as exc:
                logger.warning("MGE annotation failed: %s — skipping.", exc)
        else:
            logger.warning("MGE database not found at %s — skipping MGE annotation.", mge_db)

    # ------------------------------------------------------------------
    # 5. Mobility annotation
    # ------------------------------------------------------------------
    mobility_by_contig: dict[str, MobilityResult] = {}
    if not skip_mobility:
        logger.info("Running mob_typer on %d plasmid contigs …", len(plasmid_records))
        try:
            mob_tsv = run_mob_typer(
                plasmid_fasta,
                work_dir / "mob_typer",
                threads=threads,
            )
            mobility_results = parse_mob_results(mob_tsv)
            mobility_by_contig = index_by_contig(mobility_results)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("mob_typer unavailable or failed: %s — skipping mobility.", exc)
    else:
        logger.info("Mobility annotation skipped (skip_mobility=True)")

    # ------------------------------------------------------------------
    # 6. Taxonomy annotation (all contigs, via DIAMOND blastx against GTDB)
    # ------------------------------------------------------------------
    taxonomy_by_contig: dict[str, TaxResult] = {}
    if not skip_taxonomy:
        if taxonomy_db_path and taxonomy_db_path.exists():
            logger.info("Running taxonomy annotation on all %d contigs …", len(records))
            try:
                taxonomy_by_contig = assign_taxonomy(
                    fasta_path=fasta_path,
                    taxonomy_db=taxonomy_db_path,
                    work_dir=work_dir / "taxonomy",
                    taxon_map_path=taxon_map,
                    threads=threads,
                )
            except Exception as exc:
                logger.warning("Taxonomy annotation failed: %s — skipping.", exc)
        else:
            logger.info(
                "Taxonomy database not provided (--taxonomy-db). "
                "Use --skip-taxonomy to suppress this message."
            )
    else:
        logger.info("Taxonomy annotation skipped (skip_taxonomy=True)")

    # ------------------------------------------------------------------
    # 7. Risk scoring + assemble ContigResult list
    # ------------------------------------------------------------------
    plasmid_results: list[ContigResult] = []
    for record in plasmid_records:
        cid = record.id
        mobility = mobility_by_contig.get(cid)
        hits = args_by_contig.get(cid, [])
        risk = score_plasmid(cid, mobility, hits, source_context, taxonomy_by_contig.get(cid))
        plasmid_results.append(
            ContigResult(
                record=record,
                prediction=pred_by_id[cid],
                arg_hits=hits,
                mobility=mobility,
                risk=risk,
                taxonomy=taxonomy_by_contig.get(cid),
                vf_hits=vf_by_contig.get(cid, []),
                mge_hits=mge_by_contig.get(cid, []),
            )
        )

    # ------------------------------------------------------------------
    # 8. Build NonPlasmidContigResult list (chromosome / phage / archaea / unclassified)
    # ------------------------------------------------------------------
    non_plasmid_results: list[NonPlasmidContigResult] = []
    for record in records:
        cid = record.id
        if pred_by_id[cid].label != "plasmid":
            non_plasmid_results.append(
                NonPlasmidContigResult(
                    record=record,
                    prediction=pred_by_id[cid],
                    taxonomy=taxonomy_by_contig.get(cid),
                )
            )

    result = PipelineResult(
        input_fasta=fasta_path,
        all_predictions=predictions,
        plasmid_results=plasmid_results,
        non_plasmid_results=non_plasmid_results,
        taxonomy=taxonomy_by_contig,
    )
    tax_classified = sum(1 for r in taxonomy_by_contig.values() if r.rank != "unclassified")
    total_vf = sum(len(cr.vf_hits) for cr in plasmid_results)
    total_mge = sum(len(cr.mge_hits) for cr in plasmid_results)
    logger.info(
        "Pipeline complete — %d total | %d plasmid | %d ARGs | %d VFs | %d MGEs | "
        "%d/%d taxonomy-classified | risk scores %s",
        result.total_sequences,
        result.total_plasmids,
        result.total_args,
        total_vf,
        total_mge,
        tax_classified,
        len(taxonomy_by_contig),
        sorted({cr.risk.score for cr in plasmid_results}),
    )
    return result
