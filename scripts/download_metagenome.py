#!/usr/bin/env python3
"""Download a large assembled metagenome from NCBI for testing plasflow2.

Searches NCBI Assembly for real metagenome assemblies (wastewater, gut, soil)
that are >= a given size in MB, picks the first one with a working FTP path,
downloads it, and prints the ready-to-use plasflow2 classify command.

Usage:
    python scripts/download_metagenome.py --outdir data/test/ --min-size 150
    python scripts/download_metagenome.py --outdir data/test/ --taxon "wastewater metagenome"
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import urllib.request
from pathlib import Path

from Bio import Entrez

# NCBI requires an email for Entrez access
Entrez.email = "plasflow2@example.com"
Entrez.tool = "plasflow2"


# ---------------------------------------------------------------------------
# NCBI search
# ---------------------------------------------------------------------------

_TAXON_CHOICES = {
    "wastewater": "wastewater metagenome",
    "gut": "human gut metagenome",
    "soil": "soil metagenome",
    "any": "metagenome",
}


def search_ncbi_assemblies(taxon: str, max_results: int = 30) -> list[str]:
    """Return NCBI Assembly UIDs for assembled metagenomes of the given taxon."""
    query = f'"{taxon}"[Organism] AND "contig"[Assembly Level] AND "latest"[filter]'
    print(f"  Searching NCBI Assembly: {query}")
    handle = Entrez.esearch(db="assembly", term=query, retmax=max_results, sort="hotness")
    record = Entrez.read(handle)
    handle.close()
    ids = record.get("IdList", [])
    print(f"  Found {len(ids)} assemblies")
    return ids


def fetch_assembly_summary(uid: str) -> dict:
    """Return the Entrez document summary for a single Assembly UID."""
    handle = Entrez.esummary(db="assembly", id=uid)
    record = Entrez.read(handle, validate=False)
    handle.close()
    return record["DocumentSummarySet"]["DocumentSummary"][0]


# ---------------------------------------------------------------------------
# FTP download
# ---------------------------------------------------------------------------


def _ftp_to_https(ftp_url: str) -> str:
    return ftp_url.replace("ftp://", "https://")


def download_assembly(ftp_path: str, out_dir: Path) -> Path:
    """Download and decompress the genomic FASTA for a given assembly FTP path."""
    basename = ftp_path.rstrip("/").split("/")[-1]
    fasta_gz_name = f"{basename}_genomic.fna.gz"
    fasta_name = fasta_gz_name[:-3]  # remove .gz

    out_gz = out_dir / fasta_gz_name
    out_fasta = out_dir / fasta_name

    if out_fasta.exists():
        size_mb = out_fasta.stat().st_size / 1_000_000
        print(f"  Already downloaded: {out_fasta} ({size_mb:.1f} MB)")
        return out_fasta

    url = _ftp_to_https(f"{ftp_path}/{fasta_gz_name}")
    print(f"  Downloading: {url}")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            print(f"\r  Progress: {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, out_gz, reporthook=_progress)
    print()  # newline after progress

    print("  Decompressing …")
    with gzip.open(out_gz, "rb") as f_in, open(out_fasta, "wb") as f_out:
        while chunk := f_in.read(1 << 20):  # 1 MB chunks
            f_out.write(chunk)
    os.remove(out_gz)

    size_mb = out_fasta.stat().st_size / 1_000_000
    print(f"  Done: {out_fasta} ({size_mb:.1f} MB)")
    return out_fasta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default="data/test/",
        type=Path,
        help="Directory to save the downloaded FASTA (default: data/test/)",
    )
    parser.add_argument(
        "--min-size",
        default=150,
        type=int,
        metavar="MB",
        help="Minimum assembly size in MB (default: 150)",
    )
    parser.add_argument(
        "--taxon",
        default="wastewater metagenome",
        help=(
            "Organism/taxon to search for. "
            "Shortcuts: wastewater, gut, soil, any. "
            'Default: "wastewater metagenome"'
        ),
    )
    parser.add_argument(
        "--email",
        default="plasflow2@example.com",
        help="Email address for NCBI Entrez (required by NCBI policy).",
    )
    args = parser.parse_args()

    Entrez.email = args.email
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Expand shortcut taxon names
    taxon = _TAXON_CHOICES.get(args.taxon, args.taxon)
    min_bytes = args.min_size * 1_000_000

    print(f"Looking for '{taxon}' assemblies >= {args.min_size} MB …\n")
    uids = search_ncbi_assemblies(taxon)

    if not uids:
        print(
            f"No assemblies found for taxon '{taxon}'. "
            "Try --taxon gut, --taxon soil, or --taxon any.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Iterate candidates until we find one that meets the size threshold + has an FTP path
    downloaded: Path | None = None
    for uid in uids:
        print(f"\nChecking UID {uid} …")
        try:
            doc = fetch_assembly_summary(uid)
        except Exception as exc:
            print(f"  Entrez error: {exc} — skipping")
            continue

        accession = doc.get("AssemblyAccession", "unknown")
        organism = doc.get("Organism", "unknown")
        ftp_gb = doc.get("FtpPath_GenBank", "")
        ftp_rs = doc.get("FtpPath_RefSeq", "")
        ftp_path = ftp_rs or ftp_gb  # prefer RefSeq

        # Parse total sequence length from the Meta field (XML blob)
        meta = doc.get("Meta", "")
        total_len = 0
        import re

        m = re.search(r"<total-length>(\d+)</total-length>", meta)
        if m:
            total_len = int(m.group(1))

        size_mb = total_len / 1_000_000 if total_len else 0
        print(f"  {accession} | {organism} | {size_mb:.0f} MB")

        if not ftp_path or ftp_path == "na":
            print("  No FTP path — skipping")
            continue

        if total_len and total_len < min_bytes:
            print(f"  Too small ({size_mb:.0f} MB < {args.min_size} MB) — skipping")
            continue

        try:
            downloaded = download_assembly(ftp_path, args.outdir)
            # Double-check actual file size
            actual_mb = downloaded.stat().st_size / 1_000_000
            if actual_mb < args.min_size * 0.5:
                print(f"  File only {actual_mb:.0f} MB after download — skipping")
                downloaded.unlink()
                downloaded = None
                continue
            break
        except Exception as exc:
            print(f"  Download failed: {exc} — trying next")
            if downloaded and downloaded.exists():
                downloaded.unlink()
            downloaded = None

    if downloaded is None:
        print(
            f"\nCould not find a suitable '{taxon}' assembly >= {args.min_size} MB. "
            "Try --taxon gut or --taxon any.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Count contigs
    import subprocess

    contig_count = int(subprocess.check_output(["grep", "-c", "^>", str(downloaded)]).strip())

    out_tsv = downloaded.with_suffix("").with_suffix(".predictions.tsv")

    print(f"\n{'='*60}")
    print(f"Downloaded:   {downloaded}")
    print(f"Size:         {downloaded.stat().st_size / 1_000_000:.1f} MB")
    print(f"Contigs:      {contig_count:,}")
    print("\nRun the classifier:")
    print(
        f"  plasflow2 classify \\\n"
        f"    --input  {downloaded} \\\n"
        f"    --output {out_tsv} \\\n"
        f"    --model  data/models/mlp_v2.pt\n"
    )
    print("Then summarise:")
    print(
        f'  python3 -c "\n'
        f"import csv; from collections import Counter\n"
        f"rows = list(csv.DictReader(open('{out_tsv}'), delimiter='\\t'))\n"
        f"print(Counter(r['label'] for r in rows))\n"
        f'"'
    )


if __name__ == "__main__":
    main()
