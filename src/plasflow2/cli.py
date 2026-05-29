"""PlasFlow v2 CLI — built with Click.

Usage:
    # Full pipeline
    plasflow2 run --input assembly.fasta --output ./results/ \\
                  --threshold 0.7 --context clinical --threads 8

    # With taxonomy annotation
    plasflow2 run --input assembly.fasta --output ./results/ \\
                  --taxonomy-db data/databases/gtdb/gtdb_r220.dmnd \\
                  --taxon-map   data/databases/gtdb/taxon_map.tsv

    # Individual steps
    plasflow2 classify  --input assembly.fasta --output results/predictions.tsv
    plasflow2 annotate  --input plasmids.fasta  --output results/annotations/
    plasflow2 report    --input results/        --output results/report.html

    # Print setup / install instructions
    plasflow2 setup

Week 4 — Days 21-22 + 26 implementation.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import click

from plasflow2 import __version__
from plasflow2.annotate.args import annotate_contigs
from plasflow2.annotate.mobility import annotate_mobility
from plasflow2.classify.predict import predict
from plasflow2.pipeline import PipelineResult, run_pipeline
from plasflow2.report.generator import (
    PlasmidRow,
    _build_arg_chart,
    _build_pie_data,
    _build_risk_histogram,
    build_report_data,
    generate_report,
)
from plasflow2.risk.scorer import score_plasmid
from plasflow2.utils.fasta import load_fasta, split_by_label, write_fasta

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = Path(__file__).parent.parent.parent / "data" / "models" / "mlp_v2.pt"
_DEFAULT_CARD_DB = Path(__file__).parent.parent.parent / "data" / "databases" / "card" / "card.dmnd"
_DEFAULT_ARO_INDEX = (
    Path(__file__).parent.parent.parent / "data" / "databases" / "card" / "aro_index.tsv"
)


def _resolve_model(model_path: str | None) -> Path:
    if model_path:
        p = Path(model_path)
        if not p.exists():
            raise click.BadParameter(f"Model file not found: {p}", param_hint="--model")
        return p
    if _DEFAULT_MODEL.exists():
        return _DEFAULT_MODEL
    raise click.UsageError(
        "No model weights found. Either train a model with:\n"
        "  python scripts/train_model.py --mlp --data data/features.npy --labels data/labels.npy\n"
        "or specify --model <path>."
    )


def _write_predictions_tsv(pipeline_result: PipelineResult, output_path: Path) -> None:
    """Write comprehensive per-contig results to TSV — all classes, all annotations.

    Columns
    -------
    All contigs:
        contig_id, length, label, confidence,
        plasmid_score, chromosome_score, phage_score, archaea_score,
        taxonomy, taxonomy_rank, taxonomy_lineage

    Plasmid contigs (empty string for all other classes):
        num_args, drug_classes, arg_sources,
        mobility_class, replicon_type, relaxase_type, mpf_type,
        risk_score, mobility_score, arg_score, replicon_score,
        context_score, host_score, risk_evidence,
        eskape_host, eskape_genus
    """
    PLASMID_EMPTY = [""] * 16  # filler for non-plasmid rows

    HEADER = [
        # ── universal ────────────────────────────────────────────────────
        "contig_id",
        "length",
        "label",
        "confidence",
        "plasmid_score",
        "chromosome_score",
        "phage_score",
        "archaea_score",
        "taxonomy",
        "taxonomy_rank",
        "taxonomy_lineage",
        # ── plasmid-specific ─────────────────────────────────────────────
        "num_args",
        "drug_classes",
        "arg_sources",
        "mobility_class",
        "replicon_type",
        "relaxase_type",
        "mpf_type",
        "risk_score",
        "mobility_score",
        "arg_score",
        "replicon_score",
        "context_score",
        "host_score",
        "risk_evidence",
        "eskape_host",
        "eskape_genus",
    ]

    def _tax_fields(tax) -> list:
        if tax is None:
            return ["", "", ""]
        return [tax.display, tax.rank, tax.lineage]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Index plasmid results by contig_id for O(1) lookup
    plasmid_by_id = {cr.record.id: cr for cr in pipeline_result.plasmid_results}
    # Index all records by contig_id for length lookup
    record_by_id = {cr.record.id: cr.record for cr in pipeline_result.plasmid_results}
    record_by_id.update({cr.record.id: cr.record for cr in pipeline_result.non_plasmid_results})

    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(HEADER)

        for pred in pipeline_result.all_predictions:
            cid = pred.sequence_id
            rec = record_by_id.get(cid)
            length = len(rec.seq) if rec else 0

            scores = pred.scores if hasattr(pred, "scores") and pred.scores else {}
            base_cols = [
                cid,
                length,
                pred.label,
                f"{pred.confidence:.4f}",
                f"{scores.get('plasmid', 0):.4f}",
                f"{scores.get('chromosome', 0):.4f}",
                f"{scores.get('phage', 0):.4f}",
                f"{scores.get('archaea', 0):.4f}",
            ]

            # ── Plasmid: full annotation columns ─────────────────────────
            if pred.label == "plasmid" and cid in plasmid_by_id:
                cr = plasmid_by_id[cid]
                tax_cols = _tax_fields(cr.taxonomy)

                unique_classes = sorted(
                    {
                        dc.strip()
                        for h in cr.arg_hits
                        for dc in h.drug_class.split(";")
                        if dc.strip() and dc.strip() != "unknown"
                    }
                )
                sources = sorted(
                    {h.source for h in cr.arg_hits if hasattr(h, "source") and h.source}
                )
                mob = cr.mobility
                risk = cr.risk

                plasmid_cols = [
                    len(cr.arg_hits),
                    "; ".join(unique_classes) if unique_classes else "",
                    ", ".join(sources) if sources else "",
                    mob.mobility_class if mob else "",
                    mob.replicon_type if mob else "",
                    mob.relaxase_type if mob else "",
                    mob.mpf_type if mob else "",
                    risk.score,
                    risk.mobility_score,
                    risk.arg_score,
                    risk.replicon_score,
                    risk.context_score,
                    risk.host_score,
                    "; ".join(risk.evidence) if risk.evidence else "",
                    risk.eskape_host,
                    risk.eskape_genus,
                ]
            else:
                # Non-plasmid: taxonomy from non_plasmid_results
                tax = pipeline_result.taxonomy.get(cid)
                tax_cols = _tax_fields(tax)
                plasmid_cols = PLASMID_EMPTY

            writer.writerow(base_cols + tax_cols + plasmid_cols)


def _write_predictions_tsv_simple(predictions: list, output_path: Path) -> None:
    """Lightweight TSV writer used by 'plasflow2 classify' (no pipeline result)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                "contig_id",
                "label",
                "confidence",
                "plasmid_score",
                "chromosome_score",
                "phage_score",
                "archaea_score",
            ]
        )
        for p in predictions:
            scores = p.scores if hasattr(p, "scores") and p.scores else {}
            writer.writerow(
                [
                    p.sequence_id,
                    p.label,
                    f"{p.confidence:.4f}",
                    f"{scores.get('plasmid', 0):.4f}",
                    f"{scores.get('chromosome', 0):.4f}",
                    f"{scores.get('phage', 0):.4f}",
                    f"{scores.get('archaea', 0):.4f}",
                ]
            )


def _write_annotations_json(plasmid_results: list, output_path: Path) -> None:
    """Serialise ARG + mobility + risk + taxonomy annotations to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for cr in plasmid_results:
        mob = cr.mobility
        tax = getattr(cr, "taxonomy", None)
        records.append(
            {
                "contig_id": cr.record.id,
                "length": len(cr.record.seq),
                "classification": {
                    "label": cr.prediction.label,
                    "confidence": cr.prediction.confidence,
                },
                "taxonomy": (
                    {
                        "lineage": tax.lineage,
                        "rank": tax.rank,
                        "taxon": tax.taxon,
                        "num_hits": tax.num_hits,
                        "agreement": tax.agreement,
                    }
                    if tax
                    else None
                ),
                "mobility": (
                    {
                        "mobility_class": mob.mobility_class if mob else "unknown",
                        "replicon_type": mob.replicon_type if mob else "unknown",
                        "relaxase_type": mob.relaxase_type if mob else "none",
                        "mpf_type": mob.mpf_type if mob else "none",
                    }
                    if mob
                    else None
                ),
                "arg_hits": [
                    {
                        "gene_name": h.gene_name,
                        "aro_accession": h.aro_accession,
                        "amr_family": h.amr_family,
                        "drug_class": h.drug_class,
                        "resistance_mechanism": h.resistance_mechanism,
                        "identity": h.identity,
                        "coverage": h.coverage,
                        "evalue": h.evalue,
                        "source": h.source,
                    }
                    for h in cr.arg_hits
                ],
                "risk": {
                    "score": cr.risk.score,
                    "mobility_score": cr.risk.mobility_score,
                    "arg_score": cr.risk.arg_score,
                    "replicon_score": cr.risk.replicon_score,
                    "context_score": cr.risk.context_score,
                    "host_score": cr.risk.host_score,
                    "evidence": cr.risk.evidence,
                    "eskape_host": cr.risk.eskape_host,
                    "eskape_genus": cr.risk.eskape_genus,
                },
            }
        )
    with open(output_path, "w") as fh:
        json.dump(records, fh, indent=2)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(__version__, prog_name="plasflow2")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """PlasFlow v2 — plasmid/chromosome/phage/archaea classifier and AMR risk scorer."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _configure_logging(verbose)


# ---------------------------------------------------------------------------
# plasflow2 run  (full pipeline)
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--input",
    "-i",
    "input_fasta",
    required=True,
    type=click.Path(exists=True),
    help="Input assembly FASTA file.",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    required=True,
    type=click.Path(),
    help="Output directory (created if absent).",
)
@click.option(
    "--model",
    "model_path",
    default=None,
    type=click.Path(),
    help="Path to trained .pt weights (default: data/models/mlp_v2.pt).",
)
@click.option(
    "--card-db",
    default=None,
    type=click.Path(),
    help="DIAMOND CARD database .dmnd (default: data/databases/card/card.dmnd).",
)
@click.option(
    "--aro-index",
    default=None,
    type=click.Path(),
    help="CARD aro_index.tsv (default: data/databases/card/aro_index.tsv).",
)
@click.option(
    "--threshold",
    default=0.7,
    show_default=True,
    help="Confidence threshold for chromosome/phage/archaea; sequences below this are 'unclassified'.",
)
@click.option(
    "--plasmid-threshold",
    "plasmid_threshold",
    default=0.95,
    show_default=True,
    help=(
        "Confidence threshold for plasmid calls (default 0.95). "
        "Higher than --threshold to correct for class-prior imbalance: "
        "the model trains on ~25% plasmid but real metagenomes have ~2-5% plasmid."
    ),
)
@click.option(
    "--context",
    default="unspecified",
    show_default=True,
    type=click.Choice(
        ["clinical", "wastewater", "environmental", "unspecified"], case_sensitive=False
    ),
    help="Sample source context for AMR risk scoring.",
)
@click.option("--threads", default=8, show_default=True, help="CPU threads for DIAMOND/MOB-suite.")
@click.option("--min-length", default=1000, show_default=True, help="Minimum contig length (bp).")
@click.option(
    "--skip-mobility",
    is_flag=True,
    default=False,
    help="Skip MOB-suite mobility typing (use when mob_typer is unavailable).",
)
@click.option(
    "--taxonomy-db",
    "taxonomy_db",
    default=None,
    type=click.Path(),
    help="DIAMOND database (.dmnd) built from GTDB-r220 / RefSeq proteins for taxonomy.",
)
@click.option(
    "--taxon-map",
    "taxon_map",
    default=None,
    type=click.Path(),
    help="2-column TSV mapping accession → GTDB lineage (optional, improves LCA accuracy).",
)
@click.option(
    "--skip-taxonomy",
    is_flag=True,
    default=False,
    help="Skip taxonomy annotation (use when no taxonomy DB is available).",
)
@click.option(
    "--sarg-db",
    "sarg_db",
    default=None,
    type=click.Path(),
    help=(
        "DIAMOND database (.dmnd) built from the SARG (Structured ARG) database. "
        "When provided, ARG annotation runs against both CARD and SARG; CARD hits "
        "take precedence per ORF and SARG supplements with genes not found in CARD."
    ),
)
@click.option(
    "--min-identity",
    "min_identity",
    default=80.0,
    show_default=True,
    help=(
        "Minimum amino-acid identity %% for DIAMOND ARG hits. "
        "80%% (default) is the standard for environmental/metagenomic samples. "
        "Use 90%% for clinical-isolate-grade precision."
    ),
)
@click.option(
    "--vfdb",
    "vfdb",
    default=None,
    type=click.Path(),
    help=(
        "DIAMOND database (.dmnd) built from VFDB set A protein sequences. "
        "When provided, annotates plasmid contigs with virulence factors. "
        "Build with: diamond makedb --in VFDB_setA_pro.fas -d data/databases/vfdb/vfdb_setA"
    ),
)
@click.option(
    "--mge-db",
    "mge_db",
    default=None,
    type=click.Path(),
    help=(
        "DIAMOND database (.dmnd) built from ISfinder transposase protein sequences. "
        "When provided, detects IS elements and transposons on plasmid contigs. "
        "Build with: diamond makedb --in ISfinder-sequences.fasta -d data/databases/mge/isfinder"
    ),
)
@click.pass_context
def run(
    ctx: click.Context,
    input_fasta: str,
    output_dir: str,
    model_path: str | None,
    card_db: str | None,
    aro_index: str | None,
    threshold: float,
    plasmid_threshold: float,
    context: str,
    threads: int,
    min_length: int,
    skip_mobility: bool,
    taxonomy_db: str | None,
    taxon_map: str | None,
    skip_taxonomy: bool,
    sarg_db: str | None,
    min_identity: float,
    vfdb: str | None,
    mge_db: str | None,
) -> None:
    """Run the full PlasFlow v2 pipeline: classify → annotate → risk → report.

    \b
    Outputs written to OUTPUT_DIR:
        predictions.tsv      — per-sequence classification (all contigs)
        plasmids.fasta       — classified plasmid sequences
        annotations.json     — ARG + mobility + risk per plasmid contig
        report.html          — interactive HTML report
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    resolved_model = _resolve_model(model_path)

    card_db_path = Path(card_db) if card_db else _DEFAULT_CARD_DB
    aro_index_path = Path(aro_index) if aro_index else _DEFAULT_ARO_INDEX

    for p, name in [(card_db_path, "--card-db"), (aro_index_path, "--aro-index")]:
        if not p.exists():
            raise click.BadParameter(f"Not found: {p}", param_hint=name)

    click.echo(f"[PlasFlow v2 v{__version__}] Running pipeline on {input_fasta}")

    pipeline_result = run_pipeline(
        fasta_path=input_fasta,
        model_path=resolved_model,
        card_db=card_db_path,
        aro_index=aro_index_path,
        work_dir=out / "work",
        source_context=context,
        confidence_threshold=threshold,
        plasmid_threshold=plasmid_threshold,
        min_contig_length=min_length,
        threads=threads,
        skip_mobility=skip_mobility,
        taxonomy_db=taxonomy_db,
        taxon_map_path=taxon_map,
        skip_taxonomy=skip_taxonomy,
        sarg_db=sarg_db,
        min_identity=min_identity,
        vfdb=vfdb,
        mge_db=mge_db,
    )

    # --- Write comprehensive predictions TSV (all contigs, all annotations) ---
    preds_tsv = out / "predictions.tsv"
    _write_predictions_tsv(pipeline_result, preds_tsv)
    click.echo(f"  Predictions → {preds_tsv}")

    # --- Write per-class FASTAs (from all loaded records) ---
    records = load_fasta(input_fasta, min_length=min_length)
    pred_by_id = {p.sequence_id: p.label for p in pipeline_result.all_predictions}
    labels = [pred_by_id.get(r.id, "unclassified") for r in records]
    bins = split_by_label(records, labels)
    for label, recs in bins.items():
        fasta_out = out / f"{label}.fasta"
        write_fasta(recs, fasta_out)
        click.echo(f"  {label.capitalize()} sequences ({len(recs)}) → {fasta_out}")

    # --- Write annotations JSON ---
    ann_json = out / "annotations.json"
    _write_annotations_json(pipeline_result.plasmid_results, ann_json)
    click.echo(f"  Annotations → {ann_json}")

    # --- Write HTML report ---
    report_data = build_report_data(pipeline_result, input_file=str(input_fasta))
    report_html = out / "report.html"
    generate_report(report_data, report_html)
    click.echo(f"  Report      → {report_html}")

    click.echo(
        f"\nDone. {pipeline_result.total_sequences} sequences | "
        f"{pipeline_result.total_plasmids} plasmids | "
        f"{pipeline_result.total_args} ARGs detected."
    )


# ---------------------------------------------------------------------------
# plasflow2 classify
# ---------------------------------------------------------------------------


@main.command()
@click.option("--input", "-i", "input_fasta", required=True, type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    "output_tsv",
    required=True,
    type=click.Path(),
    help="Destination TSV file for predictions.",
)
@click.option("--model", "model_path", default=None, type=click.Path())
@click.option("--threshold", default=0.7, show_default=True)
@click.option("--min-length", "min_length", default=1000, show_default=True)
@click.pass_context
def classify(
    ctx: click.Context,
    input_fasta: str,
    output_tsv: str,
    model_path: str | None,
    threshold: float,
    min_length: int,
) -> None:
    """Classify sequences and write per-sequence predictions to TSV."""
    resolved_model = _resolve_model(model_path)

    records = load_fasta(input_fasta, min_length=min_length)
    if not records:
        click.echo(f"No sequences pass min_length={min_length} — nothing to classify.", err=True)
        return

    predictions = predict(
        [str(r.seq) for r in records],
        [r.id for r in records],
        resolved_model,
        threshold=threshold,
    )

    out_path = Path(output_tsv)
    _write_predictions_tsv_simple(predictions, out_path)

    counts: dict[str, int] = {}
    for p in predictions:
        counts[p.label] = counts.get(p.label, 0) + 1
    summary = "  ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    click.echo(f"Classified {len(predictions)} sequences — {summary}")
    click.echo(f"Predictions → {out_path}")


# ---------------------------------------------------------------------------
# plasflow2 annotate
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--input",
    "-i",
    "input_fasta",
    required=True,
    type=click.Path(exists=True),
    help="Plasmid sequences FASTA (output of 'classify' or 'run').",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    required=True,
    type=click.Path(),
    help="Output directory for intermediate files and annotations.json.",
)
@click.option("--card-db", default=None, type=click.Path())
@click.option("--aro-index", default=None, type=click.Path())
@click.option("--threads", default=8, show_default=True)
@click.option(
    "--skip-mobility",
    is_flag=True,
    default=False,
    help="Skip mob_typer mobility typing.",
)
@click.option(
    "--sarg-db",
    "sarg_db",
    default=None,
    type=click.Path(),
    help="DIAMOND database (.dmnd) built from SARG for dual-DB ARG annotation.",
)
@click.pass_context
def annotate(
    ctx: click.Context,
    input_fasta: str,
    output_dir: str,
    card_db: str | None,
    aro_index: str | None,
    threads: int,
    skip_mobility: bool,
    sarg_db: str | None,
) -> None:
    """Annotate plasmid sequences with ARGs (DIAMOND/CARD+SARG) and mobility (MOB-suite)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    card_db_path = Path(card_db) if card_db else _DEFAULT_CARD_DB
    aro_index_path = Path(aro_index) if aro_index else _DEFAULT_ARO_INDEX

    for p, name in [(card_db_path, "--card-db"), (aro_index_path, "--aro-index")]:
        if not p.exists():
            raise click.BadParameter(f"Not found: {p}", param_hint=name)

    db_label = "CARD + SARG" if sarg_db else "CARD"
    click.echo(f"Annotating ARGs on {input_fasta} ({db_label}) …")
    arg_hits = annotate_contigs(
        fasta_path=input_fasta,
        card_db=card_db_path,
        aro_index_path=aro_index_path,
        work_dir=out / "arg_work",
        threads=threads,
        sarg_db=sarg_db,
    )
    card_n = sum(1 for h in arg_hits if h.source == "CARD")
    sarg_n = sum(1 for h in arg_hits if h.source == "SARG")
    click.echo(f"  {len(arg_hits)} ARG hits detected (CARD: {card_n}, SARG: {sarg_n})")

    mobility_results = []
    if not skip_mobility:
        click.echo("Running mob_typer …")
        try:
            mobility_results = annotate_mobility(
                plasmid_fasta=input_fasta,
                work_dir=out / "mob_work",
                threads=threads,
            )
            click.echo(f"  {len(mobility_results)} mobility results")
        except (FileNotFoundError, RuntimeError) as exc:
            click.echo(f"  mob_typer unavailable: {exc} — skipping.", err=True)

    # Build a minimal annotation dict keyed by contig_id
    args_by_contig: dict[str, list] = {}
    for h in arg_hits:
        args_by_contig.setdefault(h.contig_id, []).append(h)
    mob_by_contig = {m.contig_id: m for m in mobility_results}

    all_contigs = sorted(set(args_by_contig) | set(mob_by_contig))
    records_out = []
    for cid in all_contigs:
        mob = mob_by_contig.get(cid)
        hits = args_by_contig.get(cid, [])
        records_out.append(
            {
                "contig_id": cid,
                "mobility": (
                    {
                        "mobility_class": mob.mobility_class if mob else "unknown",
                        "replicon_type": mob.replicon_type if mob else "unknown",
                        "relaxase_type": mob.relaxase_type if mob else "none",
                        "mpf_type": mob.mpf_type if mob else "none",
                    }
                    if mob
                    else None
                ),
                "arg_hits": [
                    {
                        "gene_name": h.gene_name,
                        "aro_accession": h.aro_accession,
                        "amr_family": h.amr_family,
                        "drug_class": h.drug_class,
                        "resistance_mechanism": h.resistance_mechanism,
                        "identity": h.identity,
                        "coverage": h.coverage,
                        "evalue": h.evalue,
                        "source": h.source,
                    }
                    for h in hits
                ],
            }
        )

    ann_json = out / "annotations.json"
    with open(ann_json, "w") as fh:
        json.dump(records_out, fh, indent=2)
    click.echo(f"Annotations → {ann_json}")


# ---------------------------------------------------------------------------
# plasflow2 report
# ---------------------------------------------------------------------------


@main.command("report")
@click.option(
    "--predictions",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="predictions.tsv produced by 'plasflow2 run' (comprehensive format with all annotations).",
)
@click.option(
    "--output",
    "-o",
    "output_html",
    required=True,
    type=click.Path(),
    help="Output HTML file path.",
)
@click.option(
    "--annotations",
    "-a",
    default=None,
    type=click.Path(exists=False),
    hidden=True,
    help="[Deprecated] annotations.json — ignored; all data now comes from predictions.tsv.",
)
@click.option(
    "--context",
    default="unspecified",
    type=click.Choice(
        ["clinical", "wastewater", "environmental", "unspecified"], case_sensitive=False
    ),
)
@click.pass_context
def report_cmd(
    ctx: click.Context,
    predictions: str,
    output_html: str,
    annotations: str | None,
    context: str,
) -> None:
    """Regenerate an HTML report from predictions.tsv (no pipeline re-run needed).

    The predictions.tsv produced by 'plasflow2 run' contains all annotations
    (ARGs, mobility, risk scores, taxonomy, ESKAPE host) for every contig.
    This command reads that file and rebuilds the full interactive report.
    """
    from plasflow2.annotate.args import ARGHit
    from plasflow2.annotate.mobility import MobilityResult
    from plasflow2.annotate.taxonomy import TaxResult as _TaxResult
    from plasflow2.report.generator import NonPlasmidRow

    if annotations:
        click.echo(
            "Note: --annotations is deprecated. All data is now read from predictions.tsv.",
            err=True,
        )

    plasmid_rows: list[PlasmidRow] = []
    non_plasmid_rows: list[NonPlasmidRow] = []
    all_arg_hits_for_chart: list[ARGHit] = []
    risk_scores: list[int] = []
    class_counts: dict[str, int] = {}
    total = 0

    with open(predictions) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            total += 1
            label = row.get("label", "unclassified")
            class_counts[label] = class_counts.get(label, 0) + 1
            cid = row["contig_id"]
            length = int(row.get("length", 0) or 0)
            confidence = float(row.get("confidence", 0) or 0)

            # Taxonomy
            tax_display = row.get("taxonomy", "") or "—"
            tax_rank = row.get("taxonomy_rank", "") or ""
            tax_lineage = row.get("taxonomy_lineage", "") or ""
            tax_obj: _TaxResult | None = None
            if tax_rank and tax_rank not in ("", "unclassified"):
                tax_obj = _TaxResult(
                    contig_id=cid,
                    lineage=tax_lineage,
                    rank=tax_rank,
                    taxon=row.get("taxonomy", ""),
                    num_hits=int(row.get("taxonomy_num_hits", 0) or 0),
                    agreement=float(row.get("taxonomy_agreement", 0) or 0),
                )

            if label == "plasmid":
                # Reconstruct ARGHit objects for the drug-class chart
                # (stored compactly in TSV; we rebuild from aggregate columns)
                drug_classes_str = row.get("drug_classes", "") or ""
                arg_sources_str = row.get("arg_sources", "") or ""
                num_args = int(row.get("num_args", 0) or 0)
                # Create one synthetic ARGHit per drug class for charting purposes
                for dc in drug_classes_str.split(";"):
                    dc = dc.strip()
                    if dc:
                        all_arg_hits_for_chart.append(
                            ARGHit(
                                contig_id=cid,
                                gene_name="",
                                aro_accession="",
                                amr_family="",
                                drug_class=dc,
                                resistance_mechanism="",
                                identity=100.0,
                                coverage=100.0,
                                evalue=0.0,
                                source=arg_sources_str.split(",")[0].strip() or "CARD",
                            )
                        )

                mob_class = row.get("mobility_class", "") or "unknown"
                rep_type = row.get("replicon_type", "") or "unknown"
                risk_score = int(row.get("risk_score", 0) or 0)
                risk_scores.append(risk_score)

                mob = (
                    MobilityResult(
                        contig_id=cid,
                        mobility_class=mob_class,
                        replicon_type=rep_type,
                        relaxase_type=row.get("relaxase_type", "") or "none",
                        mpf_type=row.get("mpf_type", "") or "none",
                    )
                    if mob_class
                    else None
                )

                risk = score_plasmid(cid, mob, [], context, tax_obj)
                # Override computed risk with stored values from TSV (pipeline already scored)
                risk.score = risk_score
                risk.mobility_score = int(row.get("mobility_score", 0) or 0)
                risk.arg_score = int(row.get("arg_score", 0) or 0)
                risk.replicon_score = int(row.get("replicon_score", 0) or 0)
                risk.context_score = int(row.get("context_score", 0) or 0)
                risk.host_score = int(row.get("host_score", 0) or 0)
                risk.eskape_host = row.get("eskape_host", "False").lower() == "true"
                risk.eskape_genus = row.get("eskape_genus", "") or ""
                risk.evidence = [
                    e.strip() for e in row.get("risk_evidence", "").split(";") if e.strip()
                ]

                plasmid_rows.append(
                    PlasmidRow(
                        contig_id=cid,
                        contig_length=length,
                        confidence=confidence,
                        num_args=num_args,
                        drug_classes=drug_classes_str if drug_classes_str else "—",
                        mobility_class=mob_class,
                        replicon_type=rep_type,
                        risk_score=risk_score,
                        taxonomy=tax_display,
                        risk_evidence=row.get("risk_evidence", "") or "—",
                        arg_sources=arg_sources_str,
                        eskape_host=risk.eskape_host,
                        eskape_genus=risk.eskape_genus,
                    )
                )
            else:
                non_plasmid_rows.append(
                    NonPlasmidRow(
                        contig_id=cid,
                        contig_length=length,
                        label=label,
                        confidence=confidence,
                        taxonomy=tax_display,
                        taxonomy_lineage=tax_lineage,
                    )
                )

    phage_rows = [r for r in non_plasmid_rows if r.label == "phage"]
    chromosome_rows = [r for r in non_plasmid_rows if r.label == "chromosome"]
    other_rows = [r for r in non_plasmid_rows if r.label not in ("phage", "chromosome")]

    report_data = {
        "input_file": predictions,
        "total": total,
        "num_plasmids": len(plasmid_rows),
        "total_args": sum(1 for h in all_arg_hits_for_chart),
        "class_counts": class_counts,
        "pie_data": _build_pie_data(class_counts),
        "arg_data": _build_arg_chart(all_arg_hits_for_chart),
        "risk_data": _build_risk_histogram(risk_scores),
        "plasmid_rows": plasmid_rows,
        "phage_rows": phage_rows,
        "chromosome_rows": chromosome_rows,
        "other_rows": other_rows,
        "has_phages": len(phage_rows) > 0,
        "has_chromosomes": len(chromosome_rows) > 0,
        "has_others": len(other_rows) > 0,
    }

    out_path = generate_report(report_data, output_html)
    click.echo(f"Report → {out_path}")


# ---------------------------------------------------------------------------
# plasflow2 setup
# ---------------------------------------------------------------------------

_SETUP_TEXT = """
PlasFlow v2 — External Dependency Setup
========================================

PlasFlow v2 requires the following external tools and databases.
Run the commands below once to get everything ready.

─────────────────────────────────────────
1. PYTHON DEPENDENCIES  (pip / Poetry)
─────────────────────────────────────────
    pip install poetry
    poetry install          # installs plasflow2 + all Python deps

─────────────────────────────────────────
2. SYSTEM TOOLS  (conda recommended)
─────────────────────────────────────────
    # DIAMOND  — ARG annotation + taxonomy search
    conda install -c bioconda diamond

    # MOB-suite — plasmid mobility typing
    conda install -c conda-forge -c bioconda mob_suite

    # Prodigal  — ORF prediction (Python wrapper bundled)
    # Already installed via:  pip install pyrodigal

─────────────────────────────────────────
3. CARD DATABASE  (ARG annotation)
─────────────────────────────────────────
    mkdir -p data/databases/card
    cd data/databases/card

    # Download the latest CARD data bundle:
    wget https://card.mcmaster.ca/latest/data -O card.tar.bz2

    # Extract and build DIAMOND database:
    python -c "
    from plasflow2.annotate.args import setup_card_db
    setup_card_db('data/databases/card')
    "

    # Expected output:
    #   data/databases/card/card.dmnd
    #   data/databases/card/aro_index.tsv

─────────────────────────────────────────
4. GTDB DATABASE  (taxonomy annotation)
─────────────────────────────────────────
    mkdir -p data/databases/gtdb
    cd data/databases/gtdb

    # Download GTDB-r220 representative protein sequences (~2 GB):
    wget https://data.ace.uq.edu.au/public/gtdb/data/releases/release220/220.0/\\
         genomic_files_reps/gtdb_proteins_aa_reps_r220.tar.gz

    tar xf gtdb_proteins_aa_reps_r220.tar.gz

    # Build DIAMOND protein database:
    diamond makedb \\
        --in gtdb_proteins_aa_reps_r220/gtdb_proteins_aa_reps_r220.faa \\
        -d data/databases/gtdb/gtdb_r220 \\
        --threads 8

    # Download GTDB taxonomy file and build accession→lineage map:
    wget https://data.ace.uq.edu.au/public/gtdb/data/releases/release220/220.0/\\
         bac120_taxonomy_r220.tsv.gz
    gunzip bac120_taxonomy_r220.tsv.gz

    python -c "
    from plasflow2.annotate.taxonomy import build_gtdb_taxon_map
    build_gtdb_taxon_map(
        'data/databases/gtdb/bac120_taxonomy_r220.tsv',
        'data/databases/gtdb/taxon_map.tsv'
    )
    "

    # Expected output:
    #   data/databases/gtdb/gtdb_r220.dmnd
    #   data/databases/gtdb/taxon_map.tsv

─────────────────────────────────────────
5. RUN THE FULL PIPELINE
─────────────────────────────────────────
    plasflow2 run \\
      --input      assembly.fasta \\
      --output     results/ \\
      --card-db    data/databases/card/card.dmnd \\
      --aro-index  data/databases/card/aro_index.tsv \\
      --taxonomy-db data/databases/gtdb/gtdb_r220.dmnd \\
      --taxon-map  data/databases/gtdb/taxon_map.tsv \\
      --context    wastewater \\
      --threads    8

    # Skip optional steps when databases are unavailable:
    plasflow2 run --input assembly.fasta --output results/ \\
      --skip-mobility --skip-taxonomy

─────────────────────────────────────────
6. CLASSIFY ONLY (no external databases needed)
─────────────────────────────────────────
    plasflow2 classify \\
      --input  assembly.fasta \\
      --output predictions.tsv

─────────────────────────────────────────
Tip: Run 'plasflow2 --help' for all commands and options.
"""


@main.command("setup")
def setup_cmd() -> None:
    """Print installation instructions for all external dependencies.

    Covers: Python deps, DIAMOND, MOB-suite, CARD database, GTDB database,
    and example commands for the full pipeline.
    """
    click.echo(_SETUP_TEXT)


if __name__ == "__main__":
    main()
