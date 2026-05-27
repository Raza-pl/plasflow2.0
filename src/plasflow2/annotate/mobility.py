"""MOB-suite integration for plasmid mobility and replicon typing.

Week 3 — Day 17 implementation target.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MobilityResult:
    """MOB-suite mob_typer output for one plasmid contig."""

    contig_id: str
    mobility_class: str  # conjugative | mobilizable | non-mobilizable
    replicon_type: str  # e.g., IncF, IncP, IncQ, unknown
    relaxase_type: str  # MOB family or "none"
    mpf_type: str  # Mating pair formation system or "none"


def run_mob_typer(
    plasmid_fasta: Path,
    out_dir: Path,
    threads: int = 4,
) -> Path:
    """Run MOB-suite mob_typer on classified plasmid contigs.

    Args:
        plasmid_fasta: FASTA of predicted plasmid sequences.
        out_dir: Directory where mob_typer writes its output files.
        threads: Number of CPU threads.

    Returns:
        Path to mob_typer results TSV.

    Note:
        MOB-suite may conflict with M1 conda; prefer running on CPU machine.
        See plan §2.2 and risk table.

    TODO (Day 17):
        - Handle mob_typer's non-zero exit codes for empty input gracefully.
        - Parse mobtyper_results.txt columns (field names vary by version).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mob_typer",
        "--infile",
        str(plasmid_fasta),
        "--out_file",
        str(out_dir / "mobtyper_results.txt"),
        "--num_threads",
        str(threads),
    ]
    logger.info("Running mob_typer: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_dir / "mobtyper_results.txt"


def parse_mob_results(tsv_path: Path) -> list[MobilityResult]:
    """Parse mob_typer TSV output into MobilityResult objects.

    TODO (Day 17): Map mob_typer column names to fields (version-dependent).
    """
    results: list[MobilityResult] = []
    with open(tsv_path) as fh:
        header = fh.readline().strip().split("\t")
        for line in fh:
            parts = dict(zip(header, line.strip().split("\t")))
            results.append(
                MobilityResult(
                    contig_id=parts.get("sample_id", "unknown"),
                    mobility_class=parts.get("predicted_mobility", "non-mobilizable"),
                    replicon_type=parts.get("rep_type(s)", "unknown"),
                    relaxase_type=parts.get("relaxase_type(s)", "none"),
                    mpf_type=parts.get("mpf_type", "none"),
                )
            )
    logger.info("Parsed %d mobility results from %s", len(results), tsv_path)
    return results
