#!/usr/bin/env python3
"""Download large metagenomic assembly FASTA files from NCBI for MLP training.

PlasFlow v2 classifier training benefits from real metagenome-assembled contigs
as chromosome-class examples.  Windowed RefSeq chromosomes are clean but
artificial; metagenomic assemblies capture the fragmentation, coverage depth
variation, and assembly artefacts the classifier will encounter in production.

Three metagenome contexts are downloaded (one assembly each):

    clinical      — Hospital/ICU clinical isolate metagenomes (e.g. blood,
                    sputum, wound); high AMR relevance.
    environmental — Soil, water, or ocean environmental metagenomes;
                    captures broad prokaryotic diversity.
    human_gut     — Human gut microbiome (HMP / MetaHIT studies); largest
                    and best-studied metagenomic context.

Each downloaded assembly must have >= --min-contigs contigs (default 5,000).
If an NCBI search cannot find a qualifying assembly, a curated fallback
accession is used.

All three assemblies are labeled "chromosome" in build_dataset.py.
This is a majority-label approximation: >95% of contigs in typical
metagenome assemblies are chromosomal fragments; the small plasmid/phage
fraction introduces negligible label noise.

Usage:
    # Download one qualifying assembly per metagenome type (~3–10 GB each)
    python scripts/download_metagenome_assemblies.py --outdir data/metagenomes/

    # Require at least 500,000 contigs
    python scripts/download_metagenome_assemblies.py \\
        --outdir data/metagenomes/ --min-contigs 500000

    # Dry run — search NCBI and print candidates without downloading
    python scripts/download_metagenome_assemblies.py --dry-run

    # Use NCBI API key for faster queries
    python scripts/download_metagenome_assemblies.py \\
        --outdir data/metagenomes/ --api-key YOUR_KEY_HERE

After downloading, run build_dataset.py with --metagenome-dir data/metagenomes/.
"""

from __future__ import annotations

import argparse
import gzip
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from Bio import Entrez  # type: ignore[import]

Entrez.email = "plasflow2@example.com"
Entrez.tool = "plasflow2"


# ---------------------------------------------------------------------------
# Metagenome target definitions
# ---------------------------------------------------------------------------


@dataclass
class MetagenomeTarget:
    name: str  # short identifier used for output filename
    display: str  # human-readable label for progress output
    query: str  # NCBI Assembly esearch query
    fallback_accession: str  # known-good accession if search yields nothing
    fallback_description: str


TARGETS: list[MetagenomeTarget] = [
    MetagenomeTarget(
        name="clinical",
        display="Clinical metagenome (wound/urine/hospital)",
        query=(
            '("wound metagenome"[Organism] OR "urine metagenome"[Organism] '
            'OR "sputum metagenome"[Organism] OR "clinical metagenome"[Organism]) '
            'AND "latest"[filter] NOT "chromosome"[Assembly Level]'
        ),
        # GCA_016938845.1 — wound microbiome assembly (Kalan et al.), ~18K contigs
        fallback_accession="GCA_016938845.1",
        fallback_description="Chronic wound microbiome assembly (Kalan et al. 2019)",
    ),
    MetagenomeTarget(
        name="environmental",
        display="Environmental metagenome (soil/ocean)",
        query=(
            '("soil metagenome"[Organism] OR "marine metagenome"[Organism] '
            'OR "freshwater metagenome"[Organism]) '
            'AND "latest"[filter] NOT "chromosome"[Assembly Level]'
        ),
        # GCA_002782115.1 — tundra permafrost soil metagenome, ~60K contigs
        fallback_accession="GCA_002782115.1",
        fallback_description="Tundra permafrost soil metagenome (~60K contigs)",
    ),
    MetagenomeTarget(
        name="human_gut",
        display="Human gut metagenome",
        query=(
            '("human gut metagenome"[Organism] OR "gut metagenome"[Organism] '
            'OR "human metagenome"[Organism]) '
            'AND "latest"[filter] NOT "chromosome"[Assembly Level]'
        ),
        # GCA_014875395.1 — HMP2 gut metagenome assembly, ~45K contigs
        fallback_accession="GCA_014875395.1",
        fallback_description="HMP2 human gut metagenome assembly (~45K contigs)",
    ),
]


# ---------------------------------------------------------------------------
# NCBI helpers
# ---------------------------------------------------------------------------


def _get_contig_count(summary: dict) -> int:  # type: ignore[type-arg]
    """Parse contig count from NCBI Assembly esummary Meta XML field.

    The Meta field contains embedded XML like:
        <Stats>
          <Stat category="contig_count" sequence_tag="all">412345</Stat>
          ...
        </Stats>
    """
    meta = str(summary.get("Meta", ""))
    if not meta:
        return 0
    try:
        # Meta is XML-like but may be a fragment; wrap in a root element
        root = ET.fromstring(f"<root>{meta}</root>")
        for stat in root.findall(".//Stat"):
            if stat.get("category") == "contig_count":
                return int(stat.text or 0)
    except Exception:
        pass
    # Fallback: try SequenceCount (approximate)
    try:
        return int(summary.get("SequenceCount", 0))
    except (ValueError, TypeError):
        return 0


def _get_total_length(summary: dict) -> int:  # type: ignore[type-arg]
    """Parse total assembly length from esummary Meta XML."""
    meta = str(summary.get("Meta", ""))
    if not meta:
        return 0
    try:
        root = ET.fromstring(f"<root>{meta}</root>")
        for stat in root.findall(".//Stat"):
            if stat.get("category") == "total_length":
                return int(stat.text or 0)
    except Exception:
        pass
    return 0


def _search_assemblies(query: str, max_results: int = 500, delay: float = 0.4) -> list[dict]:  # type: ignore[type-arg]
    """Search NCBI Assembly and return esummary dicts sorted by contig count desc."""
    try:
        handle = Entrez.esearch(db="assembly", term=query, retmax=max_results, sort="contig_count")
        record = Entrez.read(handle)
        handle.close()
        time.sleep(delay)
        uids = list(record.get("IdList", []))
    except Exception as exc:
        print(f"  NCBI search failed: {exc}", file=sys.stderr)
        return []

    if not uids:
        return []

    summaries: list[dict] = []  # type: ignore[type-arg]
    batch_size = 100
    for start in range(0, len(uids), batch_size):
        batch = uids[start : start + batch_size]
        try:
            handle = Entrez.esummary(db="assembly", id=",".join(batch))
            doc = Entrez.read(handle, validate=False)
            handle.close()
            summaries.extend(list(doc["DocumentSummarySet"]["DocumentSummary"]))
            time.sleep(delay)
        except Exception as exc:
            print(f"  esummary batch failed: {exc}", file=sys.stderr)

    # Sort by contig count descending so best candidates come first
    summaries.sort(key=_get_contig_count, reverse=True)
    return summaries


def _fetch_by_accession(accession: str, delay: float = 0.4) -> dict | None:  # type: ignore[type-arg]
    """Fetch assembly esummary for a specific accession."""
    try:
        handle = Entrez.esearch(db="assembly", term=f"{accession}[Assembly Accession]")
        record = Entrez.read(handle)
        handle.close()
        time.sleep(delay)
        uids = list(record.get("IdList", []))
        if not uids:
            return None
        handle = Entrez.esummary(db="assembly", id=uids[0])
        doc = Entrez.read(handle, validate=False)
        handle.close()
        time.sleep(delay)
        summaries = list(doc["DocumentSummarySet"]["DocumentSummary"])
        return summaries[0] if summaries else None
    except Exception as exc:
        print(f"  Lookup for {accession} failed: {exc}", file=sys.stderr)
        return None


def _ftp_to_https(url: str) -> str:
    return url.replace("ftp://", "https://")


def _download_and_decompress(ftp_path: str, out_fna: Path) -> bool:
    """Download genomic FASTA from *ftp_path* and decompress to *out_fna*."""
    import urllib.request

    basename = ftp_path.rstrip("/").split("/")[-1]
    gz_name = f"{basename}_genomic.fna.gz"
    url = _ftp_to_https(f"{ftp_path}/{gz_name}")
    tmp_gz = out_fna.with_suffix(".fna.gz")

    print(f"  Downloading {url}")

    def _progress(block: int, bs: int, total: int) -> None:
        if total > 0:
            mb_done = block * bs / 1_000_000
            mb_total = total / 1_000_000
            pct = min(100, block * bs * 100 // total)
            print(f"\r    {pct}%  ({mb_done:.0f}/{mb_total:.0f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, tmp_gz, reporthook=_progress)
        print()
    except Exception as exc:
        print(f"\n  Download failed: {exc}", file=sys.stderr)
        if tmp_gz.exists():
            tmp_gz.unlink()
        return False

    print(f"  Decompressing to {out_fna.name} …")
    try:
        with gzip.open(tmp_gz, "rb") as fin, open(out_fna, "wb") as fout:
            while chunk := fin.read(1 << 20):
                fout.write(chunk)
        tmp_gz.unlink()
    except Exception as exc:
        print(f"  Decompress failed: {exc}", file=sys.stderr)
        for p in (tmp_gz, out_fna):
            if p.exists():
                p.unlink()
        return False

    size_gb = out_fna.stat().st_size / 1_000_000_000
    print(f"  → {out_fna.name}  ({size_gb:.2f} GB)")
    return True


def _count_fasta_sequences(path: Path) -> int:
    """Count '>' header lines in a FASTA file as a proxy for sequence count."""
    count = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Per-target downloader
# ---------------------------------------------------------------------------


def _process_target(
    target: MetagenomeTarget,
    out_dir: Path,
    min_contigs: int,
    delay: float,
    dry_run: bool,
    verbose: bool,
) -> bool:
    """Find and download one qualifying assembly for *target*.

    Returns True if a file was downloaded (or already exists), False otherwise.
    """
    out_fna = out_dir / f"{target.name}_metagenome.fna"

    print(f"\n{'─' * 65}")
    print(f"  {target.display}")
    print(f"{'─' * 65}")

    if out_fna.exists():
        existing_count = _count_fasta_sequences(out_fna)
        print(f"  Already on disk: {out_fna.name}  ({existing_count:,} sequences)")
        if existing_count < min_contigs:
            print(
                f"  WARNING: {existing_count:,} < {min_contigs:,} required contigs. "
                f"Delete the file and re-run to replace it."
            )
        return True

    # 1. Try search-based discovery
    print(f"  Searching NCBI: {target.query}")
    summaries = _search_assemblies(target.query, max_results=500, delay=delay)
    chosen: dict | None = None  # type: ignore[type-arg]

    for s in summaries:
        contig_count = _get_contig_count(s)
        accession = str(s.get("AssemblyAccession", ""))
        total_bp = _get_total_length(s)
        ftp = str(s.get("FtpPath_RefSeq", "") or s.get("FtpPath_GenBank", ""))
        if verbose:
            print(
                f"    {accession}  contigs={contig_count:,}  "
                f"total={total_bp / 1e6:.0f} Mb  ftp={'✓' if ftp and ftp != 'na' else '✗'}"
            )
        if contig_count >= min_contigs and ftp and ftp != "na":
            chosen = s
            break

    # 2. Fall back to known accession if search failed
    if chosen is None:
        print(
            f"  Search returned no assembly with >= {min_contigs:,} contigs. "
            f"Trying fallback: {target.fallback_accession}"
        )
        chosen = _fetch_by_accession(target.fallback_accession, delay=delay)
        if chosen is None:
            print(f"  ERROR: Fallback lookup for {target.fallback_accession} also failed.")
            print(f"  ({target.fallback_description})")
            return False

    accession = str(chosen.get("AssemblyAccession", ""))
    organism = str(chosen.get("Organism", chosen.get("SpeciesName", "")))
    contig_count = _get_contig_count(chosen)
    total_bp = _get_total_length(chosen)
    ftp = str(chosen.get("FtpPath_RefSeq", "") or chosen.get("FtpPath_GenBank", ""))

    print(
        f"  Selected: {accession}  {organism}\n"
        f"           contigs={contig_count:,}  total={total_bp / 1e6:.0f} Mb"
    )

    if dry_run:
        print(f"  [dry-run] would download: {_ftp_to_https(ftp)}")
        return True

    if not ftp or ftp == "na":
        print(f"  ERROR: No FTP path for {accession}. Cannot download.")
        return False

    return _download_and_decompress(ftp, out_fna)


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
        default="data/metagenomes/",
        type=Path,
        help="Output directory for downloaded FASTA files (default: data/metagenomes/).",
    )
    parser.add_argument(
        "--min-contigs",
        type=int,
        default=5_000,
        help="Minimum contig count for a qualifying assembly (default: 5000).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        choices=["clinical", "environmental", "human_gut"],
        help="Subset of metagenome types to download (default: all three).",
    )
    parser.add_argument(
        "--email",
        default="plasflow2@example.com",
        help="Email for NCBI Entrez (required by NCBI policy).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="NCBI API key for 10 req/s rate limit (vs. 3 req/s without).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.4,
        help="Seconds between NCBI API calls (default: 0.4; auto-reduced with --api-key).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Search NCBI and print selected assemblies without downloading.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all candidate assemblies and their contig counts.",
    )
    args = parser.parse_args()

    Entrez.email = args.email
    if args.api_key:
        Entrez.api_key = args.api_key
        if args.delay >= 0.4:
            args.delay = 0.12
        print("NCBI API key set.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    active = TARGETS
    if args.targets:
        active = [t for t in TARGETS if t.name in args.targets]

    print("PlasFlow v2 — Metagenome assembly downloader")
    print(f"Targets     : {', '.join(t.name for t in active)}")
    print(f"Min contigs : {args.min_contigs:,}")
    print(f"Output dir  : {args.outdir}")
    if args.dry_run:
        print("Mode        : DRY RUN")

    results: dict[str, bool] = {}
    for target in active:
        ok = _process_target(
            target=target,
            out_dir=args.outdir,
            min_contigs=args.min_contigs,
            delay=args.delay,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        results[target.name] = ok

    # Summary
    print(f"\n{'=' * 65}")
    for name, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED"
        print(f"  {name:<20s}  {status}")
    print(f"{'=' * 65}")

    succeeded = sum(results.values())
    if succeeded == 0:
        print("\nNo assemblies downloaded. Check NCBI connectivity and try again.")
        sys.exit(1)

    if not args.dry_run:
        print("\nNext step — build training dataset:")
        print("  python scripts/build_dataset.py \\")
        print("    --plasmid-dir    data/databases/plasmidscope/ \\")
        print("    --chrom-dir      data/chromosomes/ \\")
        print(f"    --metagenome-dir {args.outdir} \\")
        print("    --data-dir       data/databases/ \\")
        print("    --max-per-class  75000 \\")
        print("    --out            data/")


if __name__ == "__main__":
    main()
