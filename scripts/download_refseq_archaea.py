#!/usr/bin/env python3
"""Download diverse RefSeq archaeal genomes for MLP classifier training.

Why archaea matter
------------------
The MLP has 4 output classes: plasmid / chromosome / phage / archaea.
Without archaeal training data the class scores near-zero probability,
and archaeal contigs get misclassified as chromosome or unclassified.

WWTP metagenomes have significant archaeal communities:
  - Methanogens (Methanosaeta, Methanosarcina): 15-30% of biomass in digesters
  - Ammonia-oxidising archaea (Nitrososphaera): 5-15% in aerobic tanks
  - Crenarchaeota, Halobacteria in sediment/biofilm

Approach
--------
Uses the NCBI FTP assembly_summary.txt — the same approach that successfully
downloaded 1998 bacterial chromosomes.  The old Entrez API approach returned
0 results because NCBI changed their API.

Usage:
    python scripts/download_refseq_archaea.py --outdir data/databases/archaea
    python scripts/download_refseq_archaea.py --outdir data/databases/archaea --count 300
"""

from __future__ import annotations

import argparse
import gzip
import logging
import random
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# NCBI FTP assembly summary for archaea (RefSeq)
ASSEMBLY_SUMMARY_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/archaea/assembly_summary.txt"
)

# Cap per genus to avoid over-representation
MAX_PER_GENUS = 10


def _fetch_assembly_summary(url: str) -> list[dict]:
    """Download and parse NCBI FTP assembly_summary.txt for archaea."""
    logger.info("Fetching assembly summary from %s …", url)
    req = urllib.request.Request(url, headers={"User-Agent": "PlasFlow2/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    lines = raw.splitlines()

    # First line is a README comment, second line is the actual header (starts with #)
    header_line = next(
        (line for line in lines if line.startswith("#") and "assembly_accession" in line),
        None,
    )
    if header_line is None:
        raise RuntimeError("Could not find header line in assembly_summary.txt")

    cols = header_line.lstrip("#").strip().split("\t")
    records = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < len(cols):
            continue
        rec = dict(zip(cols, parts))
        records.append(rec)

    logger.info("Parsed %d assembly records", len(records))
    return records


def _select_genomes(records: list[dict], total_count: int, seed: int) -> list[dict]:
    """Filter to complete/chromosome-level genomes, deduplicate by species, cap per genus."""
    # Keep only complete or chromosome-level assemblies with valid FTP paths
    filtered = [
        r for r in records
        if r.get("assembly_level") in ("Complete Genome", "Chromosome")
        and r.get("ftp_path", "na") not in ("na", "", "NA")
    ]
    logger.info("%d complete/chromosome-level archaeal assemblies available", len(filtered))

    rng = random.Random(seed)
    rng.shuffle(filtered)

    # Deduplicate: one genome per species_taxid, cap MAX_PER_GENUS per organism_name prefix
    seen_species: set[str] = set()
    genus_count: dict[str, int] = {}
    selected: list[dict] = []

    for rec in filtered:
        species_taxid = rec.get("species_taxid", "")
        if species_taxid in seen_species:
            continue

        genus = rec.get("organism_name", "Unknown").split()[0]
        if genus_count.get(genus, 0) >= MAX_PER_GENUS:
            continue

        seen_species.add(species_taxid)
        genus_count[genus] = genus_count.get(genus, 0) + 1
        selected.append(rec)

        if len(selected) >= total_count:
            break

    logger.info("Selected %d genomes after deduplication (max %d per genus)", len(selected), MAX_PER_GENUS)
    return selected


def _download_genome(rec: dict, outdir: Path) -> bool:
    """Download a single genome FASTA from NCBI FTP. Returns True on success."""
    acc = rec["assembly_accession"]
    ftp_path = rec["ftp_path"].rstrip("/")
    asm_name = rec.get("asm_name", "asm").replace(" ", "_")
    filename = f"{acc}_{asm_name}_genomic.fna.gz"
    url = f"{ftp_path}/{filename}".replace("ftp://", "https://")

    dest_fa = outdir / f"{acc}_genomic.fna"
    if dest_fa.exists():
        return True  # already downloaded

    dest_gz = outdir / f"{acc}_genomic.fna.gz"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PlasFlow2/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            dest_gz.write_bytes(resp.read())

        with gzip.open(dest_gz, "rb") as gz_in, open(dest_fa, "wb") as fa_out:
            fa_out.write(gz_in.read())
        dest_gz.unlink()
        return True

    except Exception as exc:
        logger.warning("  Failed %s: %s", acc, exc)
        dest_gz.unlink(missing_ok=True)
        return False


def download_archaea(outdir: Path, total_count: int = 200, seed: int = 42) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Check existing
    existing = list(outdir.glob("*.fna"))
    logger.info("Found %d existing archaeal genomes in %s", len(existing), outdir)

    if len(existing) >= total_count:
        logger.info("Already have %d genomes — skipping download", len(existing))
        return

    records = _fetch_assembly_summary(ASSEMBLY_SUMMARY_URL)
    targets = _select_genomes(records, total_count, seed)

    need = total_count - len(existing)
    logger.info("Downloading up to %d more genomes …", need)

    downloaded = len(existing)
    for i, rec in enumerate(targets):
        acc = rec["assembly_accession"]
        dest = outdir / f"{acc}_genomic.fna"
        if dest.exists():
            continue

        org = rec.get("organism_name", "")[:50]
        logger.info("  [%3d/%3d] %s  %s", downloaded + 1, total_count, acc, org)

        if _download_genome(rec, outdir):
            downloaded += 1

        if downloaded >= total_count:
            break

        time.sleep(0.4)  # NCBI rate limit

    final = list(outdir.glob("*.fna"))
    logger.info("Done — %d archaeal genomes in %s", len(final), outdir)
    print(f"\nDownloaded {len(final)} archaeal genomes to {outdir}/")
    print("Next step: rebuild the dataset with archaea included:")
    print(f"  python scripts/build_dataset.py \\")
    print(f"    --plasmid-dir data/databases/plasmids/ \\")
    print(f"    --chrom-dir   data/chromosomes/ \\")
    print(f"    --archaea-dir {outdir} \\")
    print(f"    --data-dir    data/databases/ \\")
    print(f"    --max-per-class 95000 \\")
    print(f"    --out         data/")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Download RefSeq archaeal genomes for PlasFlow v2 training."
    )
    parser.add_argument(
        "--outdir",
        default="data/databases/archaea",
        help="Output directory for FASTA files (default: data/databases/archaea)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Number of archaeal genomes to download (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    args = parser.parse_args()
    download_archaea(Path(args.outdir), total_count=args.count, seed=args.seed)


if __name__ == "__main__":
    main()
