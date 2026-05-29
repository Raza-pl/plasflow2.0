"""Prepare merged plasmid FASTA files from PlasmidScope tar.gz archives.

Streams each archive directly to a merged .fna file — no intermediate
extraction to disk.  Output directory contains one file per source:

    data/databases/plasmids/COMPASS.fna
    data/databases/plasmids/PLSDB.fna
    data/databases/plasmids/RefSeq.fna

Usage
-----
    python scripts/prepare_plasmid_db.py \\
        --db-dir  data/databases/ \\
        --out-dir data/databases/plasmids/

The script auto-discovers *.fasta.tar.gz files in --db-dir.
You can also specify individual archives:

    python scripts/prepare_plasmid_db.py \\
        --archives data/databases/COMPASS.fasta.tar.gz \\
                   data/databases/PLSDB.fasta.tar.gz \\
                   data/databases/RefSeq.fasta.tar.gz \\
        --out-dir  data/databases/plasmids/
"""

from __future__ import annotations

import argparse
import logging
import sys
import tarfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_FASTA_EXTS = (".fasta", ".fa", ".fna")


def extract_archive(archive_path: Path, out_path: Path) -> int:
    """Stream all FASTA entries from *archive_path* into a merged *out_path*.

    Args:
        archive_path: Path to a .tar.gz or .tgz archive of FASTA files.
        out_path: Destination merged FASTA file (overwritten if exists).

    Returns:
        Number of sequences written.
    """
    n_seq = 0
    n_files = 0
    stem = archive_path.stem.replace(".fasta", "").replace(".tar", "")
    logger.info("Processing %s → %s", archive_path.name, out_path.name)

    with tarfile.open(archive_path, "r:gz") as tar, open(out_path, "w") as out_fh:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name
            if not any(name.endswith(ext) for ext in _FASTA_EXTS):
                continue

            f = tar.extractfile(member)
            if f is None:
                continue

            n_files += 1
            for raw_line in f:
                line = raw_line.decode("utf-8", errors="replace")
                out_fh.write(line)
                if line.startswith(">"):
                    n_seq += 1

            if n_files % 10_000 == 0:
                logger.info("  … %d files / %d sequences processed", n_files, n_seq)

    logger.info("  ✓ %s: %d files → %d sequences → %s", stem, n_files, n_seq, out_path)
    return n_seq


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge PlasmidScope tar.gz archives into per-source FASTA files."
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=Path("data/databases"),
        help="Directory to auto-discover *.fasta.tar.gz files (default: data/databases/)",
    )
    parser.add_argument(
        "--archives",
        type=Path,
        nargs="+",
        help="Explicit archive paths (overrides --db-dir auto-discovery).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/databases/plasmids"),
        help="Output directory for merged FASTA files (default: data/databases/plasmids/)",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover archives
    if args.archives:
        archives = args.archives
    else:
        archives = sorted(args.db_dir.glob("*.fasta.tar.gz"))
        if not archives:
            archives = sorted(args.db_dir.glob("*.tar.gz"))

    if not archives:
        logger.error(
            "No tar.gz archives found in %s. Use --archives to specify explicitly.", args.db_dir
        )
        sys.exit(1)

    logger.info("Found %d archive(s) to process:", len(archives))
    for a in archives:
        logger.info("  %s (%.1f MB)", a.name, a.stat().st_size / 1e6)

    total_seqs = 0
    for archive in archives:
        # Derive output filename: COMPASS.fasta.tar.gz → COMPASS.fna
        stem = archive.name
        for suffix in (".fasta.tar.gz", ".tar.gz", ".fasta.tgz", ".tgz"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        out_path = out_dir / f"{stem}.fna"

        if out_path.exists():
            size_mb = out_path.stat().st_size / 1e6
            logger.info(
                "Skipping %s — output already exists (%.1f MB). Delete to re-run.",
                archive.name,
                size_mb,
            )
            # Still count sequences for summary
            with open(out_path) as fh:
                n = sum(1 for line in fh if line.startswith(">"))
            total_seqs += n
            continue

        n = extract_archive(archive, out_path)
        total_seqs += n

    logger.info("")
    logger.info("=" * 60)
    logger.info("Plasmid database prepared:")
    for f in sorted(out_dir.glob("*.fna")):
        with open(f) as fh:
            n = sum(1 for line in fh if line.startswith(">"))
        logger.info("  %-30s  %7d sequences  (%.1f MB)", f.name, n, f.stat().st_size / 1e6)
    logger.info("Total: %d sequences", total_seqs)
    logger.info("")
    logger.info("Next step:")
    logger.info("  python scripts/build_dataset.py \\")
    logger.info("    --plasmid-dir %s \\", out_dir)
    logger.info("    --data-dir data/databases/ \\")
    logger.info("    --max-per-class 75000 \\")
    logger.info("    --out data/")


if __name__ == "__main__":
    main()
