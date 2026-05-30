#!/usr/bin/env python3
"""Download diverse RefSeq complete bacterial chromosomes for MLP retraining.

Uses the NCBI FTP assembly summary file (no Entrez API, no rate limits on
the index itself) to find complete bacterial genomes, then downloads them
directly via HTTPS.

Why the old Entrez approach stopped working
-------------------------------------------
NCBI changed Entrez search behaviour so that phylum-level taxid queries
("txid1224[Organism:exp]") return 0 hits for common phyla.  The FTP
assembly_summary.txt file is updated daily and is the authoritative list.

Diversity strategy
------------------
- Reads assembly_summary.txt (~10 MB) for the bacteria RefSeq collection.
- Keeps only "Complete Genome" assemblies (fully assembled chromosomes).
- Deduplicates to ONE genome per species (species_taxid column) for maximum
  taxonomic diversity.
- Groups species by genus so you can see the phylum/class breakdown.
- Samples up to --count genomes proportionally across genera, with a
  per-genus cap to prevent one dominant genus (e.g. Salmonella) from
  swamping the dataset.

Key genera relevant to WWTP / environmental metagenomes
--------------------------------------------------------
    Proteobacteria:   Pseudomonas, Escherichia, Klebsiella, Acinetobacter,
                      Comamonas, Nitrosomonas, Nitrobacter, Burkholderiales
    Firmicutes:       Bacillus, Clostridium, Enterococcus, Staphylococcus
    Actinobacteria:   Mycobacterium, Corynebacterium, Rhodococcus
    Bacteroidota:     Bacteroides, Flavobacterium, Sphingobacterium
    Others:           Planctomycetes, Chloroflexi, Verrucomicrobia

Disk space: ~4 MB/genome × 2000 = ~8 GB

Usage:
    python scripts/download_refseq_chromosomes.py \\
        --count 2000 --outdir data/chromosomes/ --email you@example.com

    # Dry run — print what would be downloaded without fetching
    python scripts/download_refseq_chromosomes.py \\
        --count 200 --outdir data/chromosomes/ --dry-run

After downloading, rebuild and retrain:
    bash scripts/retrain_with_more_chromosomes.sh --count 2000
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import random
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ASSEMBLY_SUMMARY_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/assembly_summary.txt"

# Cap per genus — prevents one over-represented genus from dominating
MAX_PER_GENUS = 30


# ---------------------------------------------------------------------------
# Step 1: fetch and parse assembly summary
# ---------------------------------------------------------------------------


def fetch_assembly_summary() -> list[dict]:
    """Download assembly_summary.txt and return rows for Complete Genomes."""
    logger.info("Downloading NCBI RefSeq bacteria assembly summary (~10 MB)…")
    with urllib.request.urlopen(ASSEMBLY_SUMMARY_URL, timeout=60) as resp:
        raw = resp.read().decode("utf-8")

    lines = raw.splitlines()
    # First line is a comment starting with '#', second line has column headers
    # starting with '# assembly_accession' → strip leading '#'
    header_line = next(line for line in lines if line.startswith("#"))
    header = header_line.lstrip("# ").split("\t")
    data_lines = [line for line in lines if line and not line.startswith("#")]

    reader = csv.DictReader(data_lines, fieldnames=header, delimiter="\t")
    rows = []
    for row in reader:
        if (
            row.get("assembly_level") == "Complete Genome"
            and row.get("version_status", "").lower() == "latest"
            and row.get("ftp_path", "na") not in ("na", "")
        ):
            rows.append(row)

    logger.info("Complete Genome assemblies (latest): %d", len(rows))
    return rows


# ---------------------------------------------------------------------------
# Step 2: deduplicate and sample
# ---------------------------------------------------------------------------


def _genus_from_name(organism_name: str) -> str:
    """Extract genus from organism name, e.g. 'Escherichia coli K-12' → 'Escherichia'."""
    return organism_name.split()[0] if organism_name else "Unknown"


def select_assemblies(rows: list[dict], count: int, seed: int = 42) -> list[dict]:
    """Pick up to *count* assemblies, one per species, capped per genus."""
    rng = random.Random(seed)

    # Deduplicate: one assembly per species_taxid (pick randomly among available)
    by_species: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_species[row.get("species_taxid", row["taxid"])].append(row)

    species_reps: list[dict] = []
    for sp_rows in by_species.values():
        species_reps.append(rng.choice(sp_rows))

    logger.info("Unique species (one genome each): %d", len(species_reps))

    # Group by genus and apply per-genus cap
    by_genus: dict[str, list[dict]] = defaultdict(list)
    for row in species_reps:
        genus = _genus_from_name(row.get("organism_name", ""))
        by_genus[genus].append(row)

    capped: list[dict] = []
    for _genus, genus_rows in sorted(by_genus.items()):
        rng.shuffle(genus_rows)
        capped.extend(genus_rows[:MAX_PER_GENUS])

    logger.info(
        "After per-genus cap (%d): %d assemblies across %d genera",
        MAX_PER_GENUS,
        len(capped),
        len(by_genus),
    )

    # Final random sample
    rng.shuffle(capped)
    selected = capped[:count]
    logger.info("Selected %d assemblies for download", len(selected))
    return selected


# ---------------------------------------------------------------------------
# Step 3: download
# ---------------------------------------------------------------------------


def _ftp_to_https(ftp_path: str) -> str:
    """Convert ftp:// NCBI path to https:// equivalent."""
    return ftp_path.replace("ftp://", "https://", 1)


def download_genome(row: dict, out_dir: Path) -> bool:
    """Download genomic FNA for one assembly. Returns True on success."""
    acc = row["assembly_accession"]
    asm_name = row.get("asm_name", "").replace(" ", "_")
    ftp_base = _ftp_to_https(row["ftp_path"].rstrip("/"))
    # NCBI FTP convention: the genomic FASTA is named {acc}_{asm_name}_genomic.fna.gz
    fname = f"{acc}_{asm_name}_genomic.fna.gz"
    url = f"{ftp_base}/{fname}"
    dest_gz = out_dir / fname
    dest_fa = out_dir / fname.replace(".fna.gz", ".fna")

    if dest_fa.exists():
        return True  # already downloaded

    try:
        urllib.request.urlretrieve(url, str(dest_gz))
        # Decompress
        with gzip.open(dest_gz, "rb") as gz_in, open(dest_fa, "wb") as fa_out:
            fa_out.write(gz_in.read())
        dest_gz.unlink()
        return True
    except Exception as exc:
        logger.debug("Failed %s: %s", acc, exc)
        if dest_gz.exists():
            dest_gz.unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download diverse RefSeq complete bacterial chromosomes."
    )
    parser.add_argument(
        "--count", type=int, default=2000, help="Number of genomes to download (default: 2000)."
    )
    parser.add_argument(
        "--outdir",
        default="data/chromosomes/",
        help="Output directory (default: data/chromosomes/).",
    )
    parser.add_argument(
        "--email", default="plasflow2@example.com", help="Email for NCBI (informational only)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible sampling."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print selected assemblies without downloading."
    )
    parser.add_argument(
        "--delay", type=float, default=0.2, help="Seconds between downloads (default: 0.2)."
    )
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("PlasFlow v2 — RefSeq chromosome downloader (FTP summary method)")
    print(f"  Target  : {args.count} genomes")
    print(f"  Output  : {out_dir.resolve()}")
    print()

    # 1. Fetch index
    rows = fetch_assembly_summary()

    # 2. Select diverse subset
    selected = select_assemblies(rows, args.count, seed=args.seed)

    # Print genus breakdown
    by_genus: dict[str, int] = defaultdict(int)
    for row in selected:
        by_genus[_genus_from_name(row.get("organism_name", ""))] += 1
    top = sorted(by_genus.items(), key=lambda x: -x[1])[:20]
    print("Top genera in selection:")
    for genus, cnt in top:
        print(f"  {genus:<30s} {cnt:3d}")
    print(f"  … and {len(by_genus) - 20} more genera" if len(by_genus) > 20 else "")
    print()

    if args.dry_run:
        print(f"[dry-run] Would download {len(selected)} assemblies.")
        for row in selected[:20]:
            print(f"  {row['assembly_accession']}  {row.get('organism_name','')[:60]}")
        return

    # 3. Download
    already = sum(
        1
        for row in selected
        if (
            out_dir
            / f"{row['assembly_accession']}_{row.get('asm_name','').replace(' ','_')}_genomic.fna"
        ).exists()
    )
    print(f"Already on disk: {already} / {len(selected)}")

    ok_count = already
    fail_count = 0
    for row in selected:
        acc = row["assembly_accession"]
        org = row.get("organism_name", "")[:45]
        dest_fa = out_dir / f"{acc}_{row.get('asm_name','').replace(' ','_')}_genomic.fna"
        if dest_fa.exists():
            continue
        success = download_genome(row, out_dir)
        if success:
            ok_count += 1
            print(f"  [{ok_count:4d}/{len(selected)}] {acc}  {org}")
        else:
            fail_count += 1
            logger.warning("  [FAIL] %s  %s", acc, org)
        time.sleep(args.delay)

    print()
    print("=" * 60)
    print(f"  Downloaded : {ok_count}")
    print(f"  Failed     : {fail_count}")
    print(f"  Output     : {out_dir.resolve()}")
    print()
    print("Next: rebuild dataset and retrain")
    print("  bash scripts/retrain_with_more_chromosomes.sh --count 2000")
    print("=" * 60)


if __name__ == "__main__":
    main()
