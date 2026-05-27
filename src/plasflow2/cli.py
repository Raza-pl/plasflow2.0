"""PlasFlow v2 CLI — built with Click.

Usage:
    # Full pipeline
    plasflow2 run --input assembly.fasta --output ./results/ \\
                  --threshold 0.7 --context clinical --threads 8

    # Individual steps
    plasflow2 classify  --input assembly.fasta --output results/predictions.tsv
    plasflow2 annotate  --input plasmids.fasta  --output results/annotations/
    plasflow2 report    --input results/        --output results/report.html

Week 4 — Days 21-22 implementation.
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
from plasflow2.pipeline import run_pipeline
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


def _write_predictions_tsv(predictions: list, output_path: Path) -> None:
    """Write per-sequence classification results to TSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            ["sequence_id", "label", "confidence", "plasmid", "chromosome", "phage", "archaea"]
        )
        for p in predictions:
            writer.writerow(
                [
                    p.sequence_id,
                    p.label,
                    f"{p.confidence:.4f}",
                    f"{p.scores.get('plasmid', 0):.4f}",
                    f"{p.scores.get('chromosome', 0):.4f}",
                    f"{p.scores.get('phage', 0):.4f}",
                    f"{p.scores.get('archaea', 0):.4f}",
                ]
            )


def _write_annotations_json(plasmid_results: list, output_path: Path) -> None:
    """Serialise ARG + mobility + risk annotations to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for cr in plasmid_results:
        mob = cr.mobility
        records.append(
            {
                "contig_id": cr.record.id,
                "length": len(cr.record.seq),
                "classification": {
                    "label": cr.prediction.label,
                    "confidence": cr.prediction.confidence,
                },
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
                    }
                    for h in cr.arg_hits
                ],
                "risk": {
                    "score": cr.risk.score,
                    "mobility_score": cr.risk.mobility_score,
                    "arg_score": cr.risk.arg_score,
                    "replicon_score": cr.risk.replicon_score,
                    "context_score": cr.risk.context_score,
                    "evidence": cr.risk.evidence,
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
    help="Confidence threshold; sequences below this are 'unclassified'.",
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
@click.pass_context
def run(
    ctx: click.Context,
    input_fasta: str,
    output_dir: str,
    model_path: str | None,
    card_db: str | None,
    aro_index: str | None,
    threshold: float,
    context: str,
    threads: int,
    min_length: int,
    skip_mobility: bool,
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
        min_contig_length=min_length,
        threads=threads,
        skip_mobility=skip_mobility,
    )

    # --- Write predictions TSV (all contigs) ---
    preds_tsv = out / "predictions.tsv"
    _write_predictions_tsv(pipeline_result.all_predictions, preds_tsv)
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
    _write_predictions_tsv(predictions, out_path)

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
@click.pass_context
def annotate(
    ctx: click.Context,
    input_fasta: str,
    output_dir: str,
    card_db: str | None,
    aro_index: str | None,
    threads: int,
    skip_mobility: bool,
) -> None:
    """Annotate plasmid sequences with ARGs (DIAMOND/CARD) and mobility (MOB-suite)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    card_db_path = Path(card_db) if card_db else _DEFAULT_CARD_DB
    aro_index_path = Path(aro_index) if aro_index else _DEFAULT_ARO_INDEX

    for p, name in [(card_db_path, "--card-db"), (aro_index_path, "--aro-index")]:
        if not p.exists():
            raise click.BadParameter(f"Not found: {p}", param_hint=name)

    click.echo(f"Annotating ARGs on {input_fasta} …")
    arg_hits = annotate_contigs(
        fasta_path=input_fasta,
        card_db=card_db_path,
        aro_index_path=aro_index_path,
        work_dir=out / "arg_work",
        threads=threads,
    )
    click.echo(f"  {len(arg_hits)} ARG hits detected")

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
                        "drug_class": h.drug_class,
                        "identity": h.identity,
                        "coverage": h.coverage,
                        "evalue": h.evalue,
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
    "--annotations",
    "-a",
    required=True,
    type=click.Path(exists=True),
    help="annotations.json produced by 'run' or 'annotate'.",
)
@click.option(
    "--predictions",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="predictions.tsv produced by 'run' or 'classify'.",
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
    "--context",
    default="unspecified",
    type=click.Choice(
        ["clinical", "wastewater", "environmental", "unspecified"], case_sensitive=False
    ),
)
@click.pass_context
def report_cmd(
    ctx: click.Context,
    annotations: str,
    predictions: str,
    output_html: str,
    context: str,
) -> None:
    """Build an interactive HTML report from annotations + predictions TSV.

    Use this when you want to regenerate the report without re-running
    the full pipeline.
    """
    with open(annotations) as fh:
        ann_records = json.load(fh)

    # Read predictions TSV
    class_counts: dict[str, int] = {}
    total = 0
    with open(predictions) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            total += 1
            label = row.get("label", "unclassified")
            class_counts[label] = class_counts.get(label, 0) + 1

    # Build a minimal report_data dict from flat JSON
    from plasflow2.annotate.args import ARGHit
    from plasflow2.annotate.mobility import MobilityResult

    plasmid_rows = []
    all_arg_hits = []
    risk_scores = []

    for rec in ann_records:
        cid = rec["contig_id"]
        hits = [
            ARGHit(
                contig_id=cid,
                gene_name=h["gene_name"],
                aro_accession=h.get("aro_accession", "unknown"),
                amr_family=h.get("amr_family", "unknown"),
                drug_class=h["drug_class"],
                resistance_mechanism=h.get("resistance_mechanism", "unknown"),
                identity=h["identity"],
                coverage=h["coverage"],
                evalue=h["evalue"],
            )
            for h in rec.get("arg_hits", [])
        ]
        all_arg_hits.extend(hits)

        mob_data = rec.get("mobility")
        mob = (
            MobilityResult(
                contig_id=cid,
                mobility_class=mob_data["mobility_class"],
                replicon_type=mob_data["replicon_type"],
                relaxase_type=mob_data["relaxase_type"],
                mpf_type=mob_data["mpf_type"],
            )
            if mob_data
            else None
        )

        risk = score_plasmid(cid, mob, hits, context)
        risk_scores.append(risk.score)

        unique_classes = sorted(
            {
                dc.strip()
                for h in hits
                for dc in h.drug_class.split(";")
                if dc.strip() and dc.strip() != "unknown"
            }
        )
        plasmid_rows.append(
            PlasmidRow(
                contig_id=cid,
                confidence=0.0,  # not in annotations.json
                num_args=len(hits),
                drug_classes="; ".join(unique_classes) if unique_classes else "—",
                mobility_class=mob.mobility_class if mob else "unknown",
                replicon_type=mob.replicon_type if mob else "unknown",
                risk_score=risk.score,
                risk_evidence="; ".join(risk.evidence) if risk.evidence else "—",
            )
        )

    report_data = {
        "input_file": annotations,
        "total": total,
        "num_plasmids": len(ann_records),
        "total_args": len(all_arg_hits),
        "class_counts": class_counts,
        "pie_data": _build_pie_data(class_counts),
        "arg_data": _build_arg_chart(all_arg_hits),
        "risk_data": _build_risk_histogram(risk_scores),
        "plasmid_rows": plasmid_rows,
    }

    out_path = generate_report(report_data, output_html)
    click.echo(f"Report → {out_path}")


if __name__ == "__main__":
    main()
