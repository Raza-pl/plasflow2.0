"""PlasFlow v2 CLI — built with Click.

Usage:
    # Full pipeline
    plasflow2 run --input assembly.fasta --output ./results/ \\
                  --threshold 0.7 --context clinical --threads 8

    # Individual steps
    plasflow2 classify --input assembly.fasta --output preds.tsv
    plasflow2 annotate --input plasmids.fasta --output annotations/
    plasflow2 risk     --annotations annotations/ --output risk.tsv

Week 4 — Day 22 implementation target.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from plasflow2 import __version__

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
@click.option("--input",  "-i", "input_fasta",  required=True, type=click.Path(exists=True),
              help="Input assembly FASTA file.")
@click.option("--output", "-o", "output_dir",   required=True, type=click.Path(),
              help="Output directory (will be created if absent).")
@click.option("--model",        "model_path",   default=None,  type=click.Path(),
              help="Path to trained .pt weights (default: bundled model).")
@click.option("--threshold",    default=0.7,    show_default=True,
              help="Confidence threshold below which a sequence is 'unclassified'.")
@click.option("--context",      default="unspecified", show_default=True,
              type=click.Choice(["clinical", "wastewater", "environmental", "unspecified"],
                                case_sensitive=False),
              help="Sample source context for AMR risk scoring.")
@click.option("--threads",      default=8,      show_default=True,
              help="Number of CPU threads for DIAMOND/BLAST/MOB-suite.")
@click.option("--min-length",   default=1000,   show_default=True,
              help="Minimum sequence length (bp) to process.")
@click.pass_context
def run(
    ctx: click.Context,
    input_fasta: str,
    output_dir: str,
    model_path: str | None,
    threshold: float,
    context: str,
    threads: int,
    min_length: int,
) -> None:
    """Run the full PlasFlow v2 pipeline: classify → annotate → risk → report.

    Outputs written to OUTPUT_DIR:
    \b
        plasmids.fasta       — classified plasmid sequences
        chromosomes.fasta    — classified chromosomal sequences
        phages.fasta         — classified phage sequences
        archaea.fasta        — classified archaeal sequences
        unclassified.fasta   — low-confidence sequences
        predictions.tsv      — per-sequence classification results
        annotations.json     — ARG and mobility annotations
        report.html          — interactive HTML report
    """
    # TODO (Day 22): wire up the full pipeline
    click.echo(f"[PlasFlow v2 v{__version__}] Full pipeline — coming Day 22")
    click.echo(f"  Input:     {input_fasta}")
    click.echo(f"  Output:    {output_dir}")
    click.echo(f"  Threshold: {threshold}")
    click.echo(f"  Context:   {context}")
    click.echo(f"  Threads:   {threads}")


# ---------------------------------------------------------------------------
# plasflow2 classify
# ---------------------------------------------------------------------------

@main.command()
@click.option("--input",  "-i", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, type=click.Path())
@click.option("--model",        default=None,  type=click.Path())
@click.option("--threshold",    default=0.7,   show_default=True)
@click.option("--min-length",   default=1000,  show_default=True)
def classify(input: str, output: str, model: str | None, threshold: float, min_length: int) -> None:
    """Classify sequences and write predictions.tsv."""
    # TODO (Day 22): call plasflow2.classify.predict
    click.echo("classify — coming Day 22")


# ---------------------------------------------------------------------------
# plasflow2 annotate
# ---------------------------------------------------------------------------

@main.command()
@click.option("--input",  "-i", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, type=click.Path())
@click.option("--card-db",      default=None,  type=click.Path())
@click.option("--threads",      default=8,     show_default=True)
def annotate(input: str, output: str, card_db: str | None, threads: int) -> None:
    """Annotate plasmid sequences with ARGs and mobility type."""
    # TODO (Day 22): call plasflow2.annotate.args + mobility
    click.echo("annotate — coming Day 22")


# ---------------------------------------------------------------------------
# plasflow2 risk
# ---------------------------------------------------------------------------

@main.command()
@click.option("--annotations", required=True, type=click.Path(exists=True))
@click.option("--output",      required=True, type=click.Path())
@click.option("--context",     default="unspecified",
              type=click.Choice(["clinical", "wastewater", "environmental", "unspecified"],
                                case_sensitive=False))
def risk(annotations: str, output: str, context: str) -> None:
    """Compute AMR risk scores from annotation data."""
    # TODO (Day 22): call plasflow2.risk.scorer
    click.echo("risk — coming Day 22")


if __name__ == "__main__":
    main()
