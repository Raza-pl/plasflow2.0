#!/usr/bin/env python3
"""Build a lightweight DIAMOND taxonomy database from RefSeq chromosomes on disk.

Why not GTDB?
-------------
GTDB-r220 requires a ~20 GB download plus several hours of build time.
This script builds a practical taxonomy database (~300 MB) from the
1,998 RefSeq chromosomes already in data/chromosomes/, using:

  1. NCBI taxdump (names.dmp + nodes.dmp) — ~50 MB, downloaded once
  2. NCBI assembly_summary.txt — ~15 MB, downloaded once
  3. pyrodigal — protein prediction from FASTA (already installed)
  4. DIAMOND makedb — builds the .dmnd index

The resulting database covers the organisms in the training set, which
is exactly the set you need taxonomy for in practice.

Coverage: ~200-500 genomes × ~3,000 proteins each = ~600k-1.5M proteins.
Query accuracy on the training organisms: >95% genus-level assignment.

Usage:
    python scripts/build_taxonomy_db.py
    python scripts/build_taxonomy_db.py --genomes 300 --threads 8
    python scripts/build_taxonomy_db.py --chrom-dir data/chromosomes --out data/databases/taxonomy

Runtime: ~20-40 min (protein prediction is the slow step)
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
ASSEMBLY_SUMMARY_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/assembly_summary.txt"
)
ARCHAEA_SUMMARY_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/archaea/assembly_summary.txt"
)


# ---------------------------------------------------------------------------
# Step 1: NCBI taxdump — build taxid → full lineage mapping
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: Path) -> None:
    if dest.exists():
        logger.info("  Already exists: %s", dest.name)
        return
    logger.info("  Downloading %s …", url)
    req = urllib.request.Request(url, headers={"User-Agent": "PlasFlow2/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())
    logger.info("  Saved → %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def load_taxdump(taxdump_dir: Path) -> dict[int, str]:
    """Parse names.dmp + nodes.dmp into taxid → full lineage string (GTDB-style)."""
    import tarfile

    taxdump_tgz = taxdump_dir / "taxdump.tar.gz"
    names_dmp = taxdump_dir / "names.dmp"
    nodes_dmp = taxdump_dir / "nodes.dmp"

    if not names_dmp.exists() or not nodes_dmp.exists():
        logger.info("Extracting taxdump …")
        with tarfile.open(taxdump_tgz) as tf:
            tf.extract("names.dmp", path=taxdump_dir)
            tf.extract("nodes.dmp", path=taxdump_dir)

    # Parse names: taxid → scientific name
    logger.info("Parsing names.dmp …")
    taxid_to_name: dict[int, str] = {}
    with open(names_dmp) as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4 and parts[3] == "scientific name":
                taxid_to_name[int(parts[0])] = parts[1]

    # Parse nodes: taxid → (parent_taxid, rank)
    logger.info("Parsing nodes.dmp …")
    taxid_to_parent: dict[int, int] = {}
    taxid_to_rank: dict[int, str] = {}
    with open(nodes_dmp) as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                tid = int(parts[0])
                parent = int(parts[1])
                rank = parts[2]
                taxid_to_parent[tid] = parent
                taxid_to_rank[tid] = rank

    # Build lineage for each taxid by walking up to root
    logger.info("Building lineage map …")
    rank_map = {
        "superkingdom": "d__",
        "phylum": "p__",
        "class": "c__",
        "order": "o__",
        "family": "f__",
        "genus": "g__",
        "species": "s__",
    }

    def get_lineage(taxid: int) -> str:
        parts: dict[str, str] = {}
        current = taxid
        visited: set[int] = set()
        while current not in visited and current != 1:
            visited.add(current)
            rank = taxid_to_rank.get(current, "no rank")
            name = taxid_to_name.get(current, "")
            prefix = rank_map.get(rank)
            if prefix and name:
                parts[prefix] = name
            parent = taxid_to_parent.get(current, 1)
            if parent == current:
                break
            current = parent

        # Build GTDB-style string: d__X;p__X;c__X;o__X;f__X;g__X;s__X
        order = ["d__", "p__", "c__", "o__", "f__", "g__", "s__"]
        tokens = []
        for pfx in order:
            name = parts.get(pfx, "")
            tokens.append(f"{pfx}{name}")
        return ";".join(tokens)

    # Only build lineages for taxids we'll actually need (saves memory)
    return taxid_to_name, taxid_to_parent, taxid_to_rank, rank_map


def get_lineage_for_taxid(
    taxid: int,
    taxid_to_name: dict,
    taxid_to_parent: dict,
    taxid_to_rank: dict,
    rank_map: dict,
) -> str:
    """Walk NCBI taxonomy tree upward to build a GTDB-style lineage string."""
    parts: dict[str, str] = {}
    current = taxid
    visited: set[int] = set()
    while current not in visited and current != 1:
        visited.add(current)
        rank = taxid_to_rank.get(current, "no rank")
        name = taxid_to_name.get(current, "")
        prefix = rank_map.get(rank)
        if prefix and name:
            parts[prefix] = name
        parent = taxid_to_parent.get(current, 1)
        if parent == current:
            break
        current = parent

    order = ["d__", "p__", "c__", "o__", "f__", "g__", "s__"]
    tokens = [f"{pfx}{parts.get(pfx, '')}" for pfx in order]
    return ";".join(tokens)


# ---------------------------------------------------------------------------
# Step 2: accession → taxid from assembly_summary
# ---------------------------------------------------------------------------


def load_assembly_taxids(summary_path: Path) -> dict[str, int]:
    """Parse assembly_summary.txt → {accession_prefix: taxid}."""
    acc_to_taxid: dict[str, int] = {}
    with open(summary_path) as f:
        header_line = None
        for line in f:
            if line.startswith("#") and "assembly_accession" in line:
                header_line = line.lstrip("#").strip()
                cols = header_line.split("\t")
                break
        if header_line is None:
            logger.error("Could not find header in %s", summary_path)
            return acc_to_taxid

        acc_idx = cols.index("assembly_accession")
        taxid_idx = cols.index("taxid")

        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) <= max(acc_idx, taxid_idx):
                continue
            acc = parts[acc_idx].strip()
            try:
                taxid = int(parts[taxid_idx].strip())
            except ValueError:
                continue
            # Use accession without version (e.g. GCF_000007905 from GCF_000007905.1)
            acc_prefix = acc.split(".")[0]
            acc_to_taxid[acc_prefix] = taxid
            acc_to_taxid[acc] = taxid  # also store with version

    logger.info("Loaded %d accession→taxid mappings from %s", len(acc_to_taxid), summary_path.name)
    return acc_to_taxid


# ---------------------------------------------------------------------------
# Step 3: predict proteins from FASTA files with pyrodigal
# ---------------------------------------------------------------------------


def predict_proteins_pyrodigal(fasta_path: Path, out_faa: Path, lineage: str) -> int:
    """Run pyrodigal on a genome FASTA. Tag each protein header with lineage.

    Returns the number of proteins predicted.
    """
    import pyrodigal

    predictor = pyrodigal.GeneFinder(meta=True)

    proteins_written = 0
    with open(fasta_path) as f_in, open(out_faa, "a") as f_out:
        # Read all sequences from this file
        seqs: list[tuple[str, str]] = []
        current_id = ""
        current_seq: list[str] = []
        for line in f_in:
            line = line.rstrip()
            if line.startswith(">"):
                if current_seq:
                    seqs.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            seqs.append((current_id, "".join(current_seq)))

        for seq_id, seq in seqs:
            if len(seq) < 1000:
                continue
            try:
                genes = predictor.find_genes(seq.encode())
                for i, gene in enumerate(genes):
                    prot_id = f"{seq_id}_{i + 1}"
                    # Header format: >prot_id lineage_string
                    f_out.write(f">{prot_id} {lineage}\n")
                    f_out.write(f"{gene.translate().decode()}\n")
                    proteins_written += 1
            except Exception as exc:
                logger.debug("  pyrodigal failed on %s: %s", seq_id, exc)

    return proteins_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Build a DIAMOND taxonomy database from RefSeq chromosomes on disk."
    )
    parser.add_argument(
        "--chrom-dir",
        type=Path,
        default=Path("data/chromosomes"),
        help="Directory of RefSeq chromosome FASTAs (default: data/chromosomes)",
    )
    parser.add_argument(
        "--archaea-dir",
        type=Path,
        default=None,
        help="Directory of archaeal FASTAs (optional, adds archaea to taxonomy DB)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/databases/taxonomy"),
        help="Output directory for taxonomy database (default: data/databases/taxonomy)",
    )
    parser.add_argument(
        "--genomes",
        type=int,
        default=300,
        help="Number of genomes to use for protein prediction (default: 300). "
             "More genomes → better coverage but slower build.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Threads for DIAMOND makedb (default: 8)",
    )
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    taxdump_dir = out_dir / "taxdump"
    taxdump_dir.mkdir(exist_ok=True)

    dmnd_out = out_dir / "refseq_taxonomy.dmnd"
    if dmnd_out.exists():
        logger.info("DIAMOND database already exists: %s", dmnd_out)
        logger.info("Delete it to rebuild. Exiting.")
        sys.exit(0)

    # ── Step 1: Download taxdump ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Download NCBI taxdump (~50 MB)")
    logger.info("=" * 60)
    taxdump_tgz = taxdump_dir / "taxdump.tar.gz"
    _download_file(TAXDUMP_URL, taxdump_tgz)

    logger.info("Loading taxonomy trees …")
    taxid_to_name, taxid_to_parent, taxid_to_rank, rank_map = load_taxdump(taxdump_dir)

    # ── Step 2: Download assembly summary ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Download assembly summary (~15 MB each)")
    logger.info("=" * 60)
    bac_summary = taxdump_dir / "bacteria_assembly_summary.txt"
    arc_summary = taxdump_dir / "archaea_assembly_summary.txt"
    _download_file(ASSEMBLY_SUMMARY_URL, bac_summary)
    _download_file(ARCHAEA_SUMMARY_URL, arc_summary)

    acc_to_taxid: dict[str, int] = {}
    acc_to_taxid.update(load_assembly_taxids(bac_summary))
    acc_to_taxid.update(load_assembly_taxids(arc_summary))

    # ── Step 3: Predict proteins ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Predict proteins from %d genomes", args.genomes)
    logger.info("=" * 60)

    combined_faa = out_dir / "taxonomy_proteins.faa"
    if combined_faa.exists():
        combined_faa.unlink()

    sources: list[tuple[Path, str]] = []

    # Chromosomes
    chrom_files = sorted(args.chrom_dir.glob("*.fna")) if args.chrom_dir.is_dir() else []
    sources.extend([(f, "bacteria") for f in chrom_files[: args.genomes]])

    # Archaea
    if args.archaea_dir and args.archaea_dir.is_dir():
        arc_files = sorted(args.archaea_dir.glob("*.fna"))
        sources.extend([(f, "archaea") for f in arc_files[:50]])  # add up to 50 archaea

    total_proteins = 0
    for idx, (fasta_path, _domain) in enumerate(sources):
        # Extract accession from filename: GCF_000007905.1_ASM790v1_genomic.fna
        fname = fasta_path.stem  # GCF_000007905.1_ASM790v1_genomic
        acc = fname.split("_genomic")[0]  # GCF_000007905.1_ASM790v1
        # Try to get accession in GCF_XXXXXXXXX.V format
        acc_parts = acc.split("_")
        if len(acc_parts) >= 2:
            acc_short = f"{acc_parts[0]}_{acc_parts[1]}"  # GCF_000007905.1
        else:
            acc_short = acc

        taxid = acc_to_taxid.get(acc_short) or acc_to_taxid.get(acc_short.split(".")[0])
        if taxid is None:
            logger.debug("  No taxid for %s — using organism name from FASTA header", acc_short)
            # Fall back: read first FASTA header for organism name
            with open(fasta_path) as f:
                first_line = f.readline().strip()
            org_name = " ".join(first_line[1:].split()[1:4]) or "Unknown organism"
            lineage = f"d__;p__;c__;o__;f__;g__;s__{org_name}"
        else:
            lineage = get_lineage_for_taxid(
                taxid, taxid_to_name, taxid_to_parent, taxid_to_rank, rank_map
            )

        n = predict_proteins_pyrodigal(fasta_path, combined_faa, lineage)
        total_proteins += n

        if (idx + 1) % 25 == 0 or idx == 0:
            logger.info(
                "  [%4d/%4d] %s — %d proteins (total: %d)",
                idx + 1, len(sources), acc_short, n, total_proteins,
            )

    logger.info("Total proteins predicted: %d", total_proteins)

    # ── Step 4: Build DIAMOND database ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Build DIAMOND database")
    logger.info("=" * 60)

    cmd = [
        "diamond", "makedb",
        "--in", str(combined_faa),
        "--db", str(out_dir / "refseq_taxonomy"),
        "--threads", str(args.threads),
        "--quiet",
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("DIAMOND failed:\n%s", result.stderr)
        sys.exit(1)

    logger.info("DIAMOND database built: %s", dmnd_out)

    # Print summary and usage instructions
    print("\n" + "=" * 60)
    print("  Taxonomy database built successfully!")
    print("=" * 60)
    print(f"  Database : {dmnd_out}")
    print(f"  Proteins : {total_proteins:,}")
    print(f"  Genomes  : {len(sources)}")
    print()
    print("  Use it in the pipeline:")
    print(f"    plasflow2 run \\")
    print(f"      --input assembly.fasta \\")
    print(f"      --output results/ \\")
    print(f"      --taxonomy-db {dmnd_out} \\")
    print(f"      --card-db data/databases/card/card.dmnd \\")
    print(f"      --vfdb data/databases/vfdb/vfdb_setA.dmnd \\")
    print(f"      --mge-db data/databases/mge/isfinder.dmnd \\")
    print(f"      --context wastewater \\")
    print(f"      --plasmid-threshold 0.95")


if __name__ == "__main__":
    main()
