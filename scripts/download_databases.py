"""Download and verify all reference databases.

Week 1 — Day 4 implementation target.

Databases:
    PLSDB       — known plasmids (~8 GB)
    CARD        — ARG protein database (~300 MB)
    INPHARED    — phage genomes (~30k genomes)
    RefSeq      — chromosome sample (configurable subset)
    MOB-suite   — installed via conda; no manual download needed
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent / "data" / "databases"

# ---------------------------------------------------------------------------
# Database URLs and expected MD5s
# Update these when databases are versioned/updated
# ---------------------------------------------------------------------------
DATABASES: dict[str, dict] = {
    "PLSDB": {
        "url": "https://ccb-microbe.cs.uni-saarland.de/plsdb/plasmids/download/plsdb.fna.bz2",
        "dest": BASE_DIR / "plsdb" / "plsdb.fna.bz2",
        "md5": None,  # TODO: fill in after first download
        "size_hint": "~8 GB compressed",
    },
    "CARD": {
        # CARD protein homolog model FASTA
        "url": "https://card.mcmaster.ca/latest/data",
        "dest": BASE_DIR / "card" / "card.tar.bz2",
        "md5": None,  # TODO: check CARD download page for current hash
        "size_hint": "~300 MB",
    },
    "INPHARED": {
        "url": "https://millardlab-inphared.s3.climb.ac.uk/1Sep2023_phages_downloaded_from_genbank.fa.gz",
        "dest": BASE_DIR / "inphared" / "inphared_phages.fa.gz",
        "md5": None,
        "size_hint": "~30k genomes",
    },
}


def download_file(url: str, dest: Path, show_progress: bool = True) -> None:
    """Download a file with progress display."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = block_num * block_size / total_size * 100
            sys.stdout.write(f"\r  {min(pct, 100):.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress if show_progress else None)
    print()  # newline after progress


def md5_check(path: Path, expected: str) -> bool:
    """Verify file integrity against expected MD5."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        logger.error("MD5 mismatch for %s: expected %s, got %s", path, expected, actual)
        return False
    return True


def download_all(force: bool = False) -> None:
    """Download all configured databases.

    Args:
        force: Re-download even if files already exist.

    TODO (Day 4):
        - Add NCBI datasets CLI call for archaeal genomes.
        - Add NCBI datasets CLI call for RefSeq chromosome sample.
        - Build DIAMOND DB from CARD: `diamond makedb --in card.faa -d card`
        - Extract PLSDB bz2 archive.
    """
    for name, cfg in DATABASES.items():
        dest: Path = cfg["dest"]
        if dest.exists() and not force:
            logger.info("%s already present at %s — skipping", name, dest)
            continue
        logger.info("Fetching %s (%s) …", name, cfg.get("size_hint", "unknown size"))
        try:
            download_file(cfg["url"], dest)
        except Exception as exc:
            logger.error("Failed to download %s: %s", name, exc)
            continue
        if cfg.get("md5"):
            if not md5_check(dest, cfg["md5"]):
                dest.unlink()
                logger.error("Deleted corrupt file %s", dest)
            else:
                logger.info("%s verified OK", name)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download PlasFlow v2 reference databases")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()
    download_all(force=args.force)
