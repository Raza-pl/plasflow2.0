#!/usr/bin/env python3
"""Download diverse RefSeq complete bacterial chromosomes for MLP retraining.

The current PlasFlow v2 classifier was trained on chromosomal fragments from
only 35 source genomes, which causes misclassification of species not well
represented in the training set (e.g. Klebsiella pneumoniae).

This script uses NCBI taxonomy IDs to bulk-search each major bacterial phylum
for complete RefSeq assemblies, then downloads one genome per species for
maximum diversity.  Default target: 10,000 genomes across 14 phyla.

Phylum distribution (default --count 10000):
    Pseudomonadota (Proteobacteria)  3000  — largest, most clinical relevance
    Bacillota (Firmicutes)           2000
    Actinomycetota (Actinobacteria)  1500
    Bacteroidota                     1000
    Campylobacterota                  500
    Cyanobacteriota                   500
    Spirochaetota                     300
    Deinococcota                      200
    Chloroflexota                     200
    Chlamydiota                       200
    Mycoplasmatota                    200
    Fusobacteriota                    200
    Thermotogota                      100
    Aquificota + other                100

Disk space estimate:
    ~4 MB per genome (uncompressed FASTA) × 10,000 = ~40 GB
    Download bandwidth (gzipped): ~1.5 MB × 10,000 = ~15 GB transferred
    Allow at least 50 GB free space before running.

NCBI rate limits:
    Without API key: 3 requests/second → ~3 hours for 10,000 genomes
    With API key:   10 requests/second → ~1 hour for 10,000 genomes
    Register for a free NCBI API key at:
    https://www.ncbi.nlm.nih.gov/account/

Usage:
    # Download 10,000 diverse genomes (~40 GB total, ~3 hours without API key)
    python scripts/download_refseq_chromosomes.py --outdir data/chromosomes/

    # Faster with NCBI API key
    python scripts/download_refseq_chromosomes.py \\
        --outdir data/chromosomes/ --api-key YOUR_KEY_HERE

    # Smaller run for quick testing
    python scripts/download_refseq_chromosomes.py --count 200 --outdir data/chromosomes/

    # Dry run — see what would be fetched without downloading
    python scripts/download_refseq_chromosomes.py --dry-run --count 100

    # Single phylum only
    python scripts/download_refseq_chromosomes.py \\
        --phylum Pseudomonadota --count 3000 --outdir data/chromosomes/

After downloading, retrain the MLP:
    python scripts/build_dataset.py \\
        --plasmid-dir   data/databases/plasmidscope/ \\
        --chrom-dir     data/chromosomes/ \\
        --metagenome-dir data/metagenomes/ \\
        --data-dir      data/databases/ \\
        --max-per-class 75000 \\
        --out           data/

    python scripts/train_model.py \\
        --mlp --data data/features.npy --labels data/labels.npy \\
        --epochs 50 --output data/models/mlp_v2.pt
"""

from __future__ import annotations

import argparse
import gzip
import math
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from Bio import Entrez  # type: ignore[import]

Entrez.email = "plasflow2@example.com"
Entrez.tool = "plasflow2"
# Entrez.api_key is set at runtime from --api-key if provided


# ---------------------------------------------------------------------------
# Phylum definitions
# Each entry: (display_name, NCBI_taxid, fraction_of_total_count)
# Fractions sum to 1.0; actual counts are scaled by --count at runtime.
# ---------------------------------------------------------------------------


@dataclass
class Phylum:
    name: str  # display name
    taxid: int  # NCBI taxonomy ID
    fraction: float  # share of the total --count budget


PHYLA: list[Phylum] = [
    # Name                       taxid    fraction
    Phylum("Pseudomonadota", 1224, 0.30),  # Proteobacteria — huge clinical relevance
    Phylum("Bacillota", 1239, 0.20),  # Firmicutes
    Phylum("Actinomycetota", 201174, 0.15),  # Actinobacteria
    Phylum("Bacteroidota", 976, 0.10),  # Bacteroidetes
    Phylum("Campylobacterota", 29547, 0.05),  # Epsilonproteobacteria (Campylobacter/Helicobacter)
    Phylum("Cyanobacteriota", 1117, 0.05),  # Cyanobacteria
    Phylum("Spirochaetota", 203691, 0.03),  # Spirochetes
    Phylum("Deinococcota", 188787, 0.02),  # Deinococcus-Thermus
    Phylum("Chloroflexota", 200795, 0.02),  # Chloroflexi
    Phylum("Chlamydiota", 204428, 0.02),  # Chlamydiae
    Phylum("Mycoplasmatota", 2093, 0.02),  # Tenericutes / Mycoplasma
    Phylum("Fusobacteriota", 32066, 0.02),  # Fusobacteria
    Phylum("Thermotogota", 200918, 0.01),  # Thermotoga
    Phylum("Aquificota", 200783, 0.01),  # Aquificae
]
# Fractions sum to 1.0 (30+20+15+10+5+5+3+2+2+2+2+2+1+1 = 100%)


# ---------------------------------------------------------------------------
# NCBI helpers
# ---------------------------------------------------------------------------


def _search_by_taxid(taxid: int, max_results: int) -> list[str]:
    """Return up to *max_results* Assembly UIDs for *taxid* (RefSeq complete genomes).

    Searches RefSeq only (excludes GenBank-only entries) for assemblies at
    'Complete Genome' level, with the 'latest' status filter so superseded
    assemblies are excluded.
    """
    query = (
        f"txid{taxid}[Organism:exp] "
        f'AND "Complete Genome"[Assembly Level] '
        f'AND "latest"[filter] '
        f'AND "RefSeq"[Filter]'
    )
    handle = Entrez.esearch(db="assembly", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    return list(record.get("IdList", []))


def _batch_esummary(uids: list[str]) -> list[dict]:  # type: ignore[type-arg]
    """Fetch esummary for a batch of UIDs; returns list of DocumentSummary dicts."""
    if not uids:
        return []
    handle = Entrez.esummary(db="assembly", id=",".join(uids))
    doc = Entrez.read(handle, validate=False)
    handle.close()
    return list(doc["DocumentSummarySet"]["DocumentSummary"])


def _ftp_to_https(url: str) -> str:
    return url.replace("ftp://", "https://")


def _download_fasta(ftp_path: str, out_dir: Path, accession: str) -> Path | None:
    """Download and decompress the genomic FASTA for *ftp_path* into *out_dir*."""
    basename = ftp_path.rstrip("/").split("/")[-1]
    gz_name = f"{basename}_genomic.fna.gz"
    fna_name = f"{accession}_genomic.fna"
    out_fna = out_dir / fna_name

    if out_fna.exists():
        return out_fna  # already downloaded

    url = _ftp_to_https(f"{ftp_path}/{gz_name}")
    out_gz = out_dir / gz_name

    try:

        def _progress(block: int, bs: int, total: int) -> None:
            if total > 0:
                pct = min(100, block * bs * 100 // total)
                print(f"\r    {pct}%", end="", flush=True)

        urllib.request.urlretrieve(url, out_gz, reporthook=_progress)
        print()
    except Exception as exc:
        print(f"\n    Download failed: {exc}")
        if out_gz.exists():
            out_gz.unlink()
        return None

    try:
        with gzip.open(out_gz, "rb") as fin, open(out_fna, "wb") as fout:
            while chunk := fin.read(1 << 20):
                fout.write(chunk)
        out_gz.unlink()
    except Exception as exc:
        print(f"    Decompress failed: {exc}")
        for p in (out_gz, out_fna):
            if p.exists():
                p.unlink()
        return None

    return out_fna


# ---------------------------------------------------------------------------
# Per-phylum downloader
# ---------------------------------------------------------------------------


@dataclass
class PhylumResult:
    name: str
    target: int
    downloaded: list[Path] = field(default_factory=list)
    skipped: int = 0  # no FTP path
    failed: int = 0
    already: int = 0  # pre-existing on disk


def _process_phylum(
    phylum: Phylum,
    target: int,
    out_dir: Path,
    delay: float,
    dry_run: bool,
    verbose: bool,
) -> PhylumResult:
    """Fetch up to *target* genomes for *phylum*, one per species."""
    result = PhylumResult(name=phylum.name, target=target)
    if target == 0:
        return result

    print(f"\n{'─' * 60}")
    print(f"  {phylum.name}  (taxid {phylum.taxid})  target={target}")
    print(f"{'─' * 60}")

    # Fetch more UIDs than we need to have spares after deduplication.
    # NCBI retmax is capped at 10,000; we fetch up to 4× target for headroom.
    fetch_limit = min(10_000, max(target * 4, 50))

    try:
        uids = _search_by_taxid(phylum.taxid, fetch_limit)
        time.sleep(delay)
    except Exception as exc:
        print(f"  NCBI search failed: {exc}", file=sys.stderr)
        result.failed += target
        return result

    if not uids:
        print("  No RefSeq complete genomes found for this phylum.")
        return result

    if verbose:
        print(f"  Found {len(uids)} candidate assemblies")

    # Fetch summaries in batches of 200 (NCBI limit)
    summaries: list[dict] = []  # type: ignore[type-arg]
    batch_size = 200
    for start in range(0, len(uids), batch_size):
        batch = uids[start : start + batch_size]
        try:
            summaries.extend(_batch_esummary(batch))
            time.sleep(delay)
        except Exception as exc:
            print(f"  esummary batch failed: {exc}", file=sys.stderr)

    # Deduplicate: one assembly per species (keep first hit per SpeciesName)
    seen_species: set[str] = set()
    unique_summaries: list[dict] = []  # type: ignore[type-arg]
    for s in summaries:
        sp = str(s.get("SpeciesName", "")).strip().lower()
        if sp and sp not in seen_species:
            seen_species.add(sp)
            unique_summaries.append(s)

    if verbose:
        print(f"  {len(unique_summaries)} unique species after deduplication")

    downloaded_this = 0
    for s in unique_summaries:
        if downloaded_this >= target:
            break

        accession: str = str(s.get("AssemblyAccession", ""))
        organism: str = str(s.get("SpeciesName", s.get("Organism", "unknown")))
        ftp: str = str(s.get("FtpPath_RefSeq", "") or s.get("FtpPath_GenBank", ""))

        # Skip if already on disk
        existing = list(out_dir.glob(f"{accession}_genomic.fna"))
        if existing:
            print(
                f"  [{downloaded_this + 1:4d}/{target}]  {accession}  {organism}  — already on disk"
            )
            result.already += 1
            result.downloaded.extend(existing)
            downloaded_this += 1
            continue

        if not ftp or ftp == "na":
            if verbose:
                print(f"  {accession}  {organism}  — no FTP path, skipping")
            result.skipped += 1
            continue

        print(f"  [{downloaded_this + 1:4d}/{target}]  {accession}  {organism}")

        if dry_run:
            print(f"    [dry-run] {_ftp_to_https(ftp)}")
            downloaded_this += 1
            continue

        fna = _download_fasta(ftp, out_dir, accession)
        if fna:
            size_mb = fna.stat().st_size / 1_000_000
            print(f"    → {fna.name}  ({size_mb:.1f} MB)")
            result.downloaded.append(fna)
            downloaded_this += 1
        else:
            result.failed += 1

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--outdir",
        default="data/chromosomes/",
        type=Path,
        help="Directory to save downloaded FASTA files (default: data/chromosomes/).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10000,
        help=(
            "Total number of genomes to download across all phyla "
            "(default: 10000). Distributed proportionally per phylum. "
            "~40 GB disk space required for 10,000 genomes."
        ),
    )
    parser.add_argument(
        "--phylum",
        default=None,
        help=(
            "Download only genomes from this phylum "
            "(e.g. Pseudomonadota). Downloads --count genomes from that phylum only."
        ),
    )
    parser.add_argument(
        "--email",
        default="plasflow2@example.com",
        help="Email address for NCBI Entrez (required by NCBI policy).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "NCBI API key for higher rate limits (10 req/s vs 3 req/s). "
            "Register free at https://www.ncbi.nlm.nih.gov/account/ — "
            "strongly recommended for --count >= 5000."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.4,
        help="Seconds between NCBI API calls to respect rate limits (default: 0.4).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra diagnostics (candidate counts, dedup stats).",
    )
    args = parser.parse_args()

    Entrez.email = args.email
    if args.api_key:
        Entrez.api_key = args.api_key
        # With an API key NCBI allows 10 req/s; use a tighter delay
        if args.delay >= 0.4:
            args.delay = 0.12
        print(f"NCBI API key set — using {args.delay:.2f}s delay between requests.")
    elif args.count >= 5000:
        print(
            "WARNING: Downloading {:,} genomes without an NCBI API key may take 3+ hours "
            "and risks rate-limit errors.\n"
            "         Pass --api-key YOUR_KEY for 3× faster downloads.\n"
            "         Register free at https://www.ncbi.nlm.nih.gov/account/".format(args.count)
        )
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Select phyla
    active_phyla = PHYLA
    if args.phylum:
        active_phyla = [p for p in PHYLA if p.name.lower() == args.phylum.lower()]
        if not active_phyla:
            available = ", ".join(p.name for p in PHYLA)
            print(
                f"Unknown phylum '{args.phylum}'. Available: {available}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Compute per-phylum targets — proportional, at least 1 each if count allows
    total = args.count
    # Re-normalise fractions in case --phylum selected a subset
    total_fraction = sum(p.fraction for p in active_phyla)
    per_phylum_targets: dict[str, int] = {}
    allocated = 0
    for ph in active_phyla:
        share = int(math.floor(total * ph.fraction / total_fraction))
        per_phylum_targets[ph.name] = max(1, share)
        allocated += per_phylum_targets[ph.name]

    # Distribute any rounding remainder to the largest phylum
    remainder = total - allocated
    if remainder > 0 and active_phyla:
        largest = max(active_phyla, key=lambda p: p.fraction)
        per_phylum_targets[largest.name] += remainder

    print("PlasFlow v2 — RefSeq chromosome bulk downloader")
    print(f"Target    : {total} genomes")
    print(f"Output    : {args.outdir}")
    if args.dry_run:
        print("Mode      : DRY RUN (no files will be written)")
    print("\nPhylum breakdown:")
    for ph in active_phyla:
        print(f"  {ph.name:<25s}  {per_phylum_targets[ph.name]:4d}  (taxid {ph.taxid})")

    all_results: list[PhylumResult] = []

    for ph in active_phyla:
        res = _process_phylum(
            phylum=ph,
            target=per_phylum_targets[ph.name],
            out_dir=args.outdir,
            delay=args.delay,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        all_results.append(res)

    # ── Final summary ────────────────────────────────────────────────────────
    total_downloaded = sum(len(r.downloaded) for r in all_results)
    total_already = sum(r.already for r in all_results)
    total_skipped = sum(r.skipped for r in all_results)
    total_failed = sum(r.failed for r in all_results)

    print(f"\n{'=' * 60}")
    print(f"{'PHYLUM':<25s}  {'TARGET':>6}  {'GOT':>6}  {'SKIP':>5}  {'FAIL':>5}")
    print(f"{'─' * 60}")
    for r in all_results:
        print(
            f"{r.name:<25s}  {r.target:>6d}  {len(r.downloaded):>6d}"
            f"  {r.skipped:>5d}  {r.failed:>5d}"
        )
    print(f"{'─' * 60}")
    print(
        f"{'TOTAL':<25s}  {total:>6d}  {total_downloaded:>6d}"
        f"  {total_skipped:>5d}  {total_failed:>5d}"
    )
    print(f"  (of which already on disk: {total_already})")

    if not args.dry_run and total_downloaded > 0:
        new_files = [p for r in all_results for p in r.downloaded]
        total_mb = sum(p.stat().st_size for p in new_files if p.exists()) / 1_000_000
        print(f"\nTotal disk usage: {total_mb:.0f} MB in {args.outdir}")
        print("\nNext steps — retrain the MLP:")
        print("  # (Optional) Download 3 metagenome assemblies (>300K contigs each):")
        print("  python scripts/download_metagenome_assemblies.py \\")
        print("    --outdir data/metagenomes/")
        print("")
        print("  # Build training dataset:")
        print("  python scripts/build_dataset.py \\")
        print("    --plasmid-dir    data/databases/plasmidscope/ \\")
        print(f"    --chrom-dir      {args.outdir} \\")
        print("    --metagenome-dir data/metagenomes/ \\")
        print("    --data-dir       data/databases/ \\")
        print("    --max-per-class  75000 \\")
        print("    --out            data/")
        print("")
        print("  # Train MLP:")
        print("  python scripts/train_model.py \\")
        print("    --mlp --data data/features.npy --labels data/labels.npy \\")
        print("    --epochs 50 --output data/models/mlp_v2.pt")


if __name__ == "__main__":
    main()
