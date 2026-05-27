"""Build the labeled training dataset for PlasFlow v2.

Day 4 implementation.

Sources:
  - Plasmid:    data/databases/plsdb.fna         (PLSDB 2025)
  - Phage:      data/databases/inphared.fa.gz     (INPHARED Apr 2025)
  - Chromosome: downloaded via NCBI Entrez        (~50 representative RefSeq chromosomes)
  - Archaea:    data/databases/archaea.fna        (stub; full download planned Day 7)

Outputs:
  data/features.npy   — float32 array (N, 1280)
  data/labels.npy     — int64 array (N,)
  data/seq_ids.txt    — one sequence ID per line (same order)

Usage:
    python scripts/build_dataset.py [--max-per-class 10000] [--out data/]
"""

from __future__ import annotations

import argparse
import gzip
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
from Bio import Entrez, SeqIO  # type: ignore[import]

# Add src/ to path so we can import plasflow2 without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plasflow2.classify.features import extract_features  # noqa: E402
from plasflow2.utils.device import CLASS_TO_IDX  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── NCBI Entrez config ────────────────────────────────────────────────────────
# Replace with your email if you plan to run this frequently
Entrez.email = "shahbaz.invincible3182@gmail.com"

# Representative RefSeq assembly accessions for bacterial chromosomes
# Chosen to cover diverse phyla (Proteobacteria, Firmicutes, Actinobacteria,
# Bacteroidetes, Spirochaetes, Cyanobacteria) without redundancy.
CHROMOSOME_ACCESSIONS = [
    # Proteobacteria – Gammaproteobacteria
    "NC_000913",  # E. coli K-12 MG1655
    "NC_002695",  # E. coli O157:H7 Sakai
    "NC_003198",  # Salmonella Typhimurium LT2
    "NC_002516",  # Pseudomonas aeruginosa PAO1
    "NC_008463",  # Pseudomonas aeruginosa UCBPP-PA14
    "NC_003143",  # Yersinia pestis CO92
    "NC_004631",  # Salmonella Typhi CT18
    "NC_007005",  # Pseudomonas syringae pv. syringae B728a
    "NC_002940",  # Haemophilus ducreyi 35000HP
    "NC_000907",  # Haemophilus influenzae Rd KW20
    # Proteobacteria – Alphaproteobacteria
    "NC_003047",  # Sinorhizobium meliloti 1021
    "NC_007761",  # Rhizobium etli CFN 42
    "NC_002678",  # Mesorhizobium loti MAFF303099
    "NC_007618",  # Brucella melitensis biovar Abortus 2308
    "NC_003295",  # Ralstonia solanacearum GMI1000
    # Proteobacteria – Betaproteobacteria
    "NC_002927",  # Bordetella pertussis Tohama I
    "NC_003912",  # Chromobacterium violaceum ATCC 12472
    # Firmicutes
    "NC_000964",  # Bacillus subtilis 168
    "NC_002745",  # Staphylococcus aureus Mu50
    "NC_002737",  # Streptococcus pyogenes M1 GAS
    "NC_004116",  # Streptococcus agalactiae 2603V/R
    "NC_004668",  # Enterococcus faecalis V583
    "NC_003028",  # Streptococcus pneumoniae TIGR4
    "NC_009334",  # Clostridium beijerinckii NCIMB 8052
    "NC_003366",  # Clostridium perfringens str. 13
    # Actinobacteria
    "NC_000962",  # Mycobacterium tuberculosis H37Rv
    "NC_002755",  # Mycobacterium tuberculosis CDC1551
    "NC_003888",  # Streptomyces coelicolor A3(2)
    "NC_003155",  # Streptomyces avermitilis MA-4680
    "NC_003450",  # Corynebacterium glutamicum ATCC 13032
    # Bacteroidetes
    "NC_004663",  # Bacteroides thetaiotaomicron VPI-5482
    "NC_006347",  # Bacteroides fragilis NCTC 9343
    # Spirochaetes
    "NC_000117",  # Chlamydia trachomatis D/UW-3/CX
    "NC_001318",  # Borrelia burgdorferi B31
    # Cyanobacteria
    "NC_000911",  # Synechocystis sp. PCC 6803
    "NC_005070",  # Synechococcus elongatus PCC 7942
    # Thermotogae / Deinococcus-Thermus
    "NC_000853",  # Thermotoga maritima MSB8
    "NC_001263",  # Deinococcus radiodurans R1 chr 1
    # Aquificae
    "NC_000918",  # Aquifex aeolicus VF5
    # Fusobacteria
    "NC_003454",  # Fusobacterium nucleatum ATCC 25586
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _open_fasta(path: Path):
    """Return a SeqIO iterator for plain or gzipped FASTA."""
    if path.suffix == ".gz":
        return SeqIO.parse(gzip.open(path, "rt"), "fasta")
    return SeqIO.parse(str(path), "fasta")


def load_and_subsample(
    path: Path,
    label: str,
    max_per_class: int,
    min_length: int = 1000,
    seed: int = 42,
) -> tuple[list[str], list[str], list[int]]:
    """Load sequences from a FASTA, filter by length, subsample, assign label.

    Args:
        path: FASTA file (plain or .gz).
        label: Class name ('plasmid', 'chromosome', 'phage', 'archaea').
        max_per_class: Maximum number of sequences to keep.
        min_length: Minimum sequence length in bp.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (sequences, seq_ids, labels) as parallel lists.
    """
    rng = random.Random(seed)
    all_seqs: list[tuple[str, str]] = []  # (id, sequence)
    for rec in _open_fasta(path):
        seq = str(rec.seq).upper()
        if len(seq) >= min_length and set(seq) <= {"A", "C", "G", "T", "N"}:
            all_seqs.append((rec.id, seq))

    if len(all_seqs) > max_per_class:
        all_seqs = rng.sample(all_seqs, max_per_class)

    label_idx = CLASS_TO_IDX[label]
    ids = [s[0] for s in all_seqs]
    seqs = [s[1] for s in all_seqs]
    labels = [label_idx] * len(seqs)

    logger.info(
        "%-12s — kept %d sequences (min_length=%d, cap=%d)",
        label,
        len(seqs),
        min_length,
        max_per_class,
    )
    return seqs, ids, labels


def download_chromosomes(
    accessions: list[str],
    out_path: Path,
    batch_size: int = 5,
) -> Path:
    """Fetch RefSeq chromosome sequences via NCBI Entrez and save as FASTA.

    Args:
        accessions: List of RefSeq nucleotide accessions.
        out_path: Destination FASTA file.
        batch_size: Number of accessions fetched per Entrez request.

    Returns:
        Path to the written FASTA file.
    """
    if out_path.exists():
        logger.info("Chromosome FASTA already exists at %s — skipping download", out_path)
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0

    with open(out_path, "w") as fh:
        for i in range(0, len(accessions), batch_size):
            batch = accessions[i : i + batch_size]
            logger.info(
                "Fetching chromosomes %d–%d / %d: %s",
                i + 1,
                min(i + batch_size, len(accessions)),
                len(accessions),
                ", ".join(batch),
            )
            try:
                handle = Entrez.efetch(
                    db="nucleotide",
                    id=",".join(batch),
                    rettype="fasta",
                    retmode="text",
                )
                fh.write(handle.read())
                handle.close()
                total_written += len(batch)
            except Exception as exc:
                logger.warning("Failed to fetch batch %s: %s", batch, exc)
            # Be polite to NCBI — max 3 requests/second without API key
            time.sleep(0.4)

    logger.info("Downloaded %d chromosome sequences to %s", total_written, out_path)
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PlasFlow v2 training dataset")
    parser.add_argument(
        "--data-dir",
        default="data/databases",
        help="Directory containing plsdb.fna and inphared.fa.gz",
    )
    parser.add_argument(
        "--out",
        default="data",
        help="Output directory for features.npy, labels.npy, seq_ids.txt",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=10_000,
        help="Maximum sequences per class (default 10000)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1000,
        help="Minimum contig length in bp (default 1000)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip NCBI chromosome download (use existing file if present)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_seqs: list[str] = []
    all_ids: list[str] = []
    all_labels: list[int] = []

    # ── Plasmids ──────────────────────────────────────────────────────────────
    plsdb_path = data_dir / "plsdb.fna"
    if not plsdb_path.exists():
        # Try alternate filename from download script
        plsdb_path = data_dir / "sequences.fasta"
    if plsdb_path.exists():
        seqs, ids, labels = load_and_subsample(
            plsdb_path, "plasmid", args.max_per_class, args.min_length
        )
        all_seqs.extend(seqs)
        all_ids.extend(ids)
        all_labels.extend(labels)
    else:
        logger.warning("PLSDB not found at %s — skipping plasmid class", plsdb_path)

    # ── Phages ────────────────────────────────────────────────────────────────
    inphared_path = data_dir / "inphared.fa.gz"
    if not inphared_path.exists():
        inphared_path = data_dir / "14Apr2025_genomes.fa.gz"
    if inphared_path.exists():
        seqs, ids, labels = load_and_subsample(
            inphared_path, "phage", args.max_per_class, args.min_length
        )
        all_seqs.extend(seqs)
        all_ids.extend(ids)
        all_labels.extend(labels)
    else:
        logger.warning("INPHARED not found at %s — skipping phage class", inphared_path)

    # ── Chromosomes ───────────────────────────────────────────────────────────
    chr_path = data_dir / "chromosomes.fna"
    if not args.skip_download:
        download_chromosomes(CHROMOSOME_ACCESSIONS, chr_path)
    if chr_path.exists():
        seqs, ids, labels = load_and_subsample(
            chr_path, "chromosome", args.max_per_class, args.min_length
        )
        all_seqs.extend(seqs)
        all_ids.extend(ids)
        all_labels.extend(labels)
    else:
        logger.warning("Chromosome FASTA not found — skipping chromosome class")

    # ── Archaea (stub) ────────────────────────────────────────────────────────
    # Full archaeal download is planned for Day 7.  If the file exists, use it;
    # otherwise skip (the class will be absent from this training run).
    archaea_path = data_dir / "archaea.fna"
    if archaea_path.exists():
        seqs, ids, labels = load_and_subsample(
            archaea_path, "archaea", args.max_per_class, args.min_length
        )
        all_seqs.extend(seqs)
        all_ids.extend(ids)
        all_labels.extend(labels)
    else:
        logger.info("Archaea FASTA not found — class will be absent (stub until Day 7)")

    # ── Feature extraction ────────────────────────────────────────────────────
    if not all_seqs:
        logger.error("No sequences loaded — check database paths and re-run.")
        sys.exit(1)

    logger.info("Extracting k-mer features for %d sequences …", len(all_seqs))
    X = extract_features(all_seqs)
    y = np.array(all_labels, dtype=np.int64)

    # ── Save outputs ──────────────────────────────────────────────────────────
    feat_path = out_dir / "features.npy"
    label_path = out_dir / "labels.npy"
    ids_path = out_dir / "seq_ids.txt"

    np.save(str(feat_path), X)
    np.save(str(label_path), y)
    ids_path.write_text("\n".join(all_ids))

    logger.info("Saved features  → %s  shape=%s", feat_path, X.shape)
    logger.info("Saved labels    → %s  shape=%s", label_path, y.shape)
    logger.info("Saved seq IDs   → %s  (%d lines)", ids_path, len(all_ids))

    # ── Class summary ─────────────────────────────────────────────────────────
    from plasflow2.utils.device import IDX_TO_CLASS

    logger.info("\nClass distribution:")
    for idx, name in IDX_TO_CLASS.items():
        count = int((y == idx).sum())
        logger.info("  %-12s  %6d sequences", name, count)


if __name__ == "__main__":
    main()
