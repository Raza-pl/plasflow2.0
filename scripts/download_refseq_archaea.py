#!/usr/bin/env python3
"""Download diverse RefSeq complete archaeal genomes for MLP classifier training.

Why archaea matter in PlasFlow v2
----------------------------------
The MLP has 4 output classes: plasmid / chromosome / phage / archaea.
Currently the archaea class has ZERO training sequences, so:
  - The class always predicts with near-zero probability.
  - Archaeal contigs get misclassified as chromosome (closest k-mer profile)
    or left as unclassified, which inflates chromosome counts.

Adding archaeal training data allows the model to:
  1. Correctly identify archaeal contigs (prevents false positives in the
     chromosome / phage classes).
  2. Suppress archaea from AMR annotation (archaea don't carry clinically
     relevant resistance genes — skipping them is intentional).

Archaea in environmental metagenomes
-------------------------------------
WWTP (wastewater treatment plants) — the primary PlasFlow v2 use case — have
significant archaeal communities:
  - Anaerobic digesters:  methanogens (Methanosaeta, Methanobacterium,
    Methanosarcina) often represent 15–30 % of microbial biomass.
  - Aerobic tanks:        Thaumarchaeota (ammonia-oxidising archaea like
    Nitrososphaera) can reach 5–15 %.
  - Sediment/biofilm:    Crenarchaeota, Halobacteria in high-salinity WW.

Without archaeal training data, these abundant sequences pollute the
chromosome class and reduce the model's effective accuracy on the non-archaea
classes.

Archaeal phyla and target counts (default --count 500)
--------------------------------------------------------
    Euryarchaeota      200  — methanogens, halophiles (most WWTP-relevant)
    Thermoprotei       100  — Crenarchaeota / hyperthermophiles
    Thaumarchaeota      80  — ammonia-oxidising archaea (WWTP nitrification)
    Asgard archaea      50  — recently discovered, diverse
    Nanoarchaeota       30  — ultra-small symbiotic archaea
    DPANN group         40  — diverse, tiny genomes

Disk space: ~2 MB/genome (archaea have smaller genomes than bacteria) × 500
            = ~1 GB.

Usage:
    python scripts/download_refseq_archaea.py \\
        --out-dir data/databases/archaea \\
        --count 500

    # Then add to build_dataset.py:
    archaea_dir = Path("data/databases/archaea")
    # load_windowed_streaming handles it identically to chromosome sequences
"""

from __future__ import annotations

import argparse
import gzip
import logging
import random
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# NCBI taxonomy IDs for archaeal phyla / groups
ARCHAEA_TAXIDS: dict[str, tuple[int, int]] = {
    # (NCBI taxonomy ID for phylum/class, target count)
    "Euryarchaeota": (28890, 200),
    "Crenarchaeota": (28889, 100),
    "Thaumarchaeota": (651137, 80),
    "Asgard_archaea": (1935183, 50),
    "Nanoarchaeota": (192989, 30),
    "DPANN": (1783276, 40),
}


def _fetch_assembly_list(taxid: int, max_results: int = 500) -> list[dict[str, str]]:
    """Query NCBI Datasets API for RefSeq complete archaeal assemblies."""
    import json
    import urllib.error

    url = (
        f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/taxon/{taxid}/dataset_report"
        f"?filters.assembly_source=refseq"
        f"&filters.assembly_level=complete_genome"
        f"&filters.exclude_atypical=true"
        f"&page_size={min(max_results, 1000)}"
        f"&returned_content=ASSEMBLIES"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        logger.warning("NCBI API call failed for taxid %d: %s", taxid, exc)
        return []

    assemblies = []
    for report in data.get("reports", []):
        acc = report.get("accession", "")
        name = report.get("organism", {}).get("organism_name", "Unknown")
        ftp = report.get("assembly_info", {}).get("ftp_path_genbank", "") or report.get(
            "assembly_info", {}
        ).get("ftp_path", "")
        if acc and ftp:
            assemblies.append({"accession": acc, "name": name, "ftp": ftp})
    return assemblies


def _ftp_download(ftp_path: str, dest: Path) -> bool:
    """Download the genomic FASTA from an NCBI FTP path. Returns True on success."""
    # ftp_path looks like: https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/.../
    if not ftp_path:
        return False
    ftp_path = ftp_path.rstrip("/")
    basename = ftp_path.split("/")[-1]
    url = f"{ftp_path}/{basename}_genomic.fna.gz"
    try:
        logger.debug("Downloading %s", url)
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception as exc:
        logger.warning("Download failed for %s: %s", url, exc)
        return False


def download_archaea(
    out_dir: Path,
    total_count: int = 500,
    api_key: str | None = None,
    seed: int = 42,
) -> None:
    """Download archaeal genomes from RefSeq, distributed across phyla.

    Args:
        out_dir: Directory to write downloaded FASTA files.
        total_count: Total number of genomes to download.
        api_key: NCBI API key (optional, increases rate limit from 3 to 10 req/s).
        seed: Random seed for reproducible sampling.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Apportion targets proportionally across phyla
    total_weights = sum(v[1] for v in ARCHAEA_TAXIDS.values())
    targets: dict[str, int] = {
        name: max(1, round(total_count * count / total_weights))
        for name, (_, count) in ARCHAEA_TAXIDS.items()
    }
    # Adjust rounding errors
    diff = total_count - sum(targets.values())
    if diff:
        largest = max(targets, key=targets.get)  # type: ignore[arg-type]
        targets[largest] += diff

    logger.info("Download targets: %s", targets)
    downloaded = 0

    for phylum, (taxid, _) in ARCHAEA_TAXIDS.items():
        want = targets[phylum]
        logger.info("Fetching assembly list for %s (taxid=%d, want=%d) …", phylum, taxid, want)
        assemblies = _fetch_assembly_list(taxid, max_results=want * 3)
        if not assemblies:
            logger.warning("No assemblies found for %s — skipping", phylum)
            continue

        rng.shuffle(assemblies)
        got = 0
        for asm in assemblies:
            if got >= want:
                break
            acc = asm["accession"]
            dest_gz = out_dir / f"{acc}_genomic.fna.gz"
            dest_fa = out_dir / f"{acc}_genomic.fna"

            if dest_fa.exists():
                logger.info("  [skip] %s already exists", acc)
                got += 1
                downloaded += 1
                continue

            if _ftp_download(asm["ftp"], dest_gz):
                # Decompress
                try:
                    with gzip.open(dest_gz, "rb") as gz_in, open(dest_fa, "wb") as fa_out:
                        fa_out.write(gz_in.read())
                    dest_gz.unlink()
                    logger.info("  [%3d] %-40s  %s", downloaded + 1, asm["name"][:40], acc)
                    got += 1
                    downloaded += 1
                except Exception as exc:
                    logger.warning("  Decompress failed for %s: %s", acc, exc)
                    dest_gz.unlink(missing_ok=True)

            # NCBI rate limiting
            time.sleep(0.35 if api_key else 1.0)

        logger.info("  %s: downloaded %d / %d", phylum, got, want)

    logger.info("Total archaeal genomes downloaded: %d / %d", downloaded, total_count)
    logger.info("Output directory: %s", out_dir)

    # Print next steps
    print("\nNext steps:")
    print("  1. Add archaea sequences to build_dataset.py:")
    print(f"     archaea_dir = Path('{out_dir}')")
    print("     archaea_files = list(archaea_dir.glob('*.fna'))")
    print("     archaea_seqs = load_windowed_streaming(archaea_files, label='archaea', ...)")
    print("  2. Rebuild the dataset and retrain:")
    print("     python scripts/build_dataset.py \\")
    print("       --plasmid-dir data/databases/PlasmidScope \\")
    print(f"       --archaea-dir {out_dir} \\")
    print("       --output data/datasets/training_v3.npz")
    print("  3. python -m plasflow2.classify.train --dataset data/datasets/training_v3.npz")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download RefSeq archaeal genomes for PlasFlow v2 training."
    )
    parser.add_argument(
        "--out-dir",
        default="data/databases/archaea",
        help="Output directory for downloaded FASTA files (default: data/databases/archaea).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Total number of archaeal genomes to download (default: 500).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="NCBI API key for higher rate limits (optional). "
        "Register at: https://www.ncbi.nlm.nih.gov/account/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    args = parser.parse_args()

    download_archaea(
        out_dir=Path(args.out_dir),
        total_count=args.count,
        api_key=args.api_key,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
