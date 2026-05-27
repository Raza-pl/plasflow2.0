"""MOB-suite integration for plasmid mobility and replicon typing.

Week 3 — Day 17 implementation.

Pipeline:
    plasmid FASTA → run_mob_typer() → TSV → parse_mob_results() → [MobilityResult]

MOB-suite installation (one-time, conda recommended):
    conda install -c bioconda mob_suite

Note: MOB-suite may conflict with Apple Silicon conda envs.
Prefer running on a Linux/x86 machine or via Docker.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# mob_typer column names (mob_suite >= 3.0)
# ---------------------------------------------------------------------------

# Canonical column names in mobtyper_results.txt (mob_suite 3.x)
_COL_SAMPLE_ID = "sample_id"
_COL_MOBILITY = "predicted_mobility"
_COL_REP_TYPE = "rep_type(s)"
_COL_RELAXASE = "relaxase_type(s)"
_COL_MPF = "mpf_type"

# Valid mobility classes returned by mob_typer
MOBILITY_CLASSES = frozenset({"conjugative", "mobilizable", "non-mobilizable"})


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class MobilityResult:
    """MOB-suite mob_typer output for one plasmid contig."""

    contig_id: str
    mobility_class: str  # conjugative | mobilizable | non-mobilizable
    replicon_type: str  # e.g., IncF, IncP, IncQ, unknown
    relaxase_type: str  # MOB family or "none"
    mpf_type: str  # Mating pair formation system or "none"
    # Raw row dict preserved for downstream callers that need other fields
    raw: dict[str, str] = field(default_factory=dict, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Running mob_typer
# ---------------------------------------------------------------------------


def run_mob_typer(
    plasmid_fasta: Path | str,
    out_dir: Path | str,
    threads: int = 4,
) -> Path:
    """Run MOB-suite mob_typer on classified plasmid contigs.

    Args:
        plasmid_fasta: FASTA of predicted plasmid sequences.
        out_dir: Directory where mob_typer writes its output files.
        threads: Number of CPU threads.

    Returns:
        Path to mob_typer results TSV (mobtyper_results.txt).

    Raises:
        FileNotFoundError: If mob_typer is not on PATH.
        RuntimeError: If mob_typer exits non-zero and the results file
                      is also absent (i.e., a real failure, not just
                      empty input).
    """
    plasmid_fasta = Path(plasmid_fasta)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_tsv = out_dir / "mobtyper_results.txt"

    cmd = [
        "mob_typer",
        "--infile",
        str(plasmid_fasta),
        "--out_file",
        str(results_tsv),
        "--num_threads",
        str(threads),
    ]
    logger.info("Running mob_typer: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # mob_typer exits non-zero for empty / unclassifiable input but may
        # still produce a header-only results file — treat that as success.
        if results_tsv.exists():
            logger.warning(
                "mob_typer exited %d but produced results file — continuing. " "stderr: %s",
                result.returncode,
                result.stderr.strip(),
            )
        else:
            logger.error("mob_typer stderr: %s", result.stderr.strip())
            raise RuntimeError(
                f"mob_typer failed (exit {result.returncode}) and produced no "
                f"results file. See log for stderr."
            )

    return results_tsv


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_mob_results(tsv_path: Path | str) -> list[MobilityResult]:
    """Parse mob_typer TSV output into MobilityResult objects.

    Handles mob_suite >= 3.0 column names. Unknown / missing columns are
    filled with sensible defaults so the parser does not crash on minor
    version differences.

    Args:
        tsv_path: Path to mobtyper_results.txt produced by run_mob_typer().

    Returns:
        List of MobilityResult, one per data row.  Returns an empty list
        for header-only files (no contigs classified).
    """
    tsv_path = Path(tsv_path)
    results: list[MobilityResult] = []

    with open(tsv_path) as fh:
        raw_header = fh.readline()
        if not raw_header:
            logger.info("mob_typer results file is empty: %s", tsv_path)
            return results

        header = [h.strip() for h in raw_header.split("\t")]

        for line in fh:
            line = line.strip()
            if not line:
                continue
            values = line.split("\t")
            row = dict(zip(header, values, strict=False))

            # Normalise mobility class — default to non-mobilizable if absent
            mob_class = row.get(_COL_MOBILITY, "non-mobilizable").strip().lower()
            if mob_class not in MOBILITY_CLASSES:
                logger.debug(
                    "Unrecognised mobility class %r — defaulting to non-mobilizable",
                    mob_class,
                )
                mob_class = "non-mobilizable"

            # Normalise replicon: strip whitespace; "-" or empty -> "unknown"
            rep_type = row.get(_COL_REP_TYPE, "-").strip()
            if rep_type in ("-", ""):
                rep_type = "unknown"

            # Normalise relaxase / MPF
            relaxase = row.get(_COL_RELAXASE, "-").strip()
            if relaxase in ("-", ""):
                relaxase = "none"

            mpf = row.get(_COL_MPF, "-").strip()
            if mpf in ("-", ""):
                mpf = "none"

            results.append(
                MobilityResult(
                    contig_id=row.get(_COL_SAMPLE_ID, "unknown").strip(),
                    mobility_class=mob_class,
                    replicon_type=rep_type,
                    relaxase_type=relaxase,
                    mpf_type=mpf,
                    raw=row,
                )
            )

    logger.info("Parsed %d mobility results from %s", len(results), tsv_path)
    return results


# ---------------------------------------------------------------------------
# Convenience: full mobility annotation for one FASTA
# ---------------------------------------------------------------------------


def annotate_mobility(
    plasmid_fasta: Path | str,
    work_dir: Path | str,
    threads: int = 4,
) -> list[MobilityResult]:
    """End-to-end mobility annotation: mob_typer -> parsed results.

    Args:
        plasmid_fasta: FASTA of predicted plasmid sequences.
        work_dir: Directory for mob_typer intermediate files.
        threads: CPU threads for mob_typer.

    Returns:
        List of MobilityResult across all contigs.  Empty list if
        mob_typer produces no results (e.g., all contigs too short).
    """
    results_tsv = run_mob_typer(plasmid_fasta, work_dir, threads=threads)
    return parse_mob_results(results_tsv)


# ---------------------------------------------------------------------------
# Index helper
# ---------------------------------------------------------------------------


def index_by_contig(results: list[MobilityResult]) -> dict[str, MobilityResult]:
    """Return a dict mapping contig_id -> MobilityResult for fast lookup."""
    return {r.contig_id: r for r in results}
