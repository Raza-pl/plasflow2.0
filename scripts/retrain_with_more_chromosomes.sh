#!/usr/bin/env bash
# =============================================================================
# retrain_with_more_chromosomes.sh
#
# Downloads diverse RefSeq chromosomes AND archaeal genomes, rebuilds the
# 4-class training dataset (plasmid/chromosome/phage/archaea), and retrains
# the MLP classifier.
#
# Problems being fixed
# --------------------
# 1. Chromosome class: trained on 40 genomes → 73% of WWTP contigs unclassified.
#    Fix: 1,998 diverse RefSeq chromosomes downloaded; retrain with 95k windows.
#
# 2. Archaea class: ZERO training examples → archaea scored near 0 probability.
#    Fix: download 200 diverse archaeal genomes via RefSeq FTP; include in training.
#    WWTP metagenomes have 15-30% archaea in anaerobic digesters — this matters.
#
# Runtime estimate:
#   Chromosome download : already done (1,998 in data/chromosomes/)
#   Archaea download    : ~15 min (200 genomes × ~1 MB each)
#   Dataset build       : ~20 min (380k windows across 4 classes)
#   Retrain             : ~20 min (50 epochs, early stopping, CPU)
#   Total               : ~1 h
#
# Usage:
#   bash scripts/retrain_with_more_chromosomes.sh
#   bash scripts/retrain_with_more_chromosomes.sh --count 2000  # more chromosomes
#   bash scripts/retrain_with_more_chromosomes.sh --archaea 300  # more archaea
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CHROM_DIR="$PROJECT_DIR/data/chromosomes"
DATA_DIR="$PROJECT_DIR/data"
MODEL_OUT="$DATA_DIR/models/mlp_v2.pt"
MODEL_BAK="$DATA_DIR/models/mlp_v2_backup_$(date +%Y%m%d_%H%M%S).pt"

GENOME_COUNT=1000
ARCHAEA_COUNT=200
API_KEY=""
ARCHAEA_DIR="$DATA_DIR/databases/archaea"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --count)    GENOME_COUNT="$2"; shift 2 ;;
        --archaea)  ARCHAEA_COUNT="$2"; shift 2 ;;
        --api-key)  API_KEY="$2";      shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  PlasFlow v2 — Full 4-class retraining pipeline"
echo "  Bacterial chromosomes : $GENOME_COUNT  (in $CHROM_DIR)"
echo "  Archaeal genomes      : $ARCHAEA_COUNT (in $ARCHAEA_DIR)"
echo "  Model output          : $MODEL_OUT"
echo "============================================================"
echo ""

# ── 0. Activate conda env ────────────────────────────────────────────────────
# Try to activate plasflow2 env; if already active, no-op
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate plasflow2 2>/dev/null || true
fi

cd "$PROJECT_DIR"

# ── 1. Backup existing model ──────────────────────────────────────────────────
if [[ -f "$MODEL_OUT" ]]; then
    echo "[backup] Saving existing model → $MODEL_BAK"
    cp "$MODEL_OUT" "$MODEL_BAK"
fi

# ── 2. Download chromosomes (skip if already present) ────────────────────────
echo ""
echo "━━━ Step 1/4: Bacterial chromosomes ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ALREADY=$(find "$CHROM_DIR" -name "*.fna" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$ALREADY" -ge "$GENOME_COUNT" ]]; then
    echo "[✓] Already have $ALREADY genomes in $CHROM_DIR — skipping"
else
    echo "[info] Found $ALREADY — downloading to $GENOME_COUNT …"
    python scripts/download_refseq_chromosomes.py \
        --count "$GENOME_COUNT" \
        --outdir "$CHROM_DIR"
    echo "[✓] Chromosomes ready: $(find "$CHROM_DIR" -name "*.fna" | wc -l | tr -d ' ') files"
fi

# ── 3. Download archaeal genomes ──────────────────────────────────────────────
echo ""
echo "━━━ Step 2/4: Archaeal genomes ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Target: $ARCHAEA_COUNT genomes — methanogens, Thaumarchaeota, Crenarchaeota"
echo "  Output: $ARCHAEA_DIR/"
mkdir -p "$ARCHAEA_DIR"

ARC_ALREADY=$(find "$ARCHAEA_DIR" -name "*.fna" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$ARC_ALREADY" -ge "$ARCHAEA_COUNT" ]]; then
    echo "[✓] Already have $ARC_ALREADY archaeal genomes — skipping"
else
    echo "[info] Found $ARC_ALREADY — downloading to $ARCHAEA_COUNT …"
    python scripts/download_refseq_archaea.py \
        --outdir "$ARCHAEA_DIR" \
        --count "$ARCHAEA_COUNT"
    echo "[✓] Archaea ready: $(find "$ARCHAEA_DIR" -name "*.fna" | wc -l | tr -d ' ') files"
fi

# ── 4. Rebuild training dataset ───────────────────────────────────────────────
echo ""
echo "━━━ Step 3/4: Rebuild training dataset (4 classes) ━━━━━━━━━━━━━━━━━━━━━"
echo "  Plasmid   — data/databases/plasmids/   (PLSDB + RefSeq + COMPASS)"
echo "  Chromosome— $CHROM_DIR/ ($ALREADY genomes, tiled to windows)"
echo "  Phage     — data/databases/inphared/   (INPHARED)"
echo "  Archaea   — $ARCHAEA_DIR/ ($ARC_ALREADY genomes, tiled to windows)"
echo ""

python scripts/build_dataset.py \
    --plasmid-dir  data/databases/plasmids/ \
    --chrom-dir    "$CHROM_DIR" \
    --archaea-dir  "$ARCHAEA_DIR" \
    --data-dir     data/databases/ \
    --max-per-class 95000 \
    --out          data/

echo ""
python3 - <<'PYEOF'
import numpy as np, collections
labels = np.load("data/labels.npy")
counts = dict(sorted(collections.Counter(labels.tolist()).items()))
names = {0: "plasmid", 1: "chromosome", 2: "phage", 3: "archaea"}
print("  Dataset label distribution:")
for k, v in counts.items():
    print(f"    {names.get(k, str(k)):<12s}: {v:,}")
print(f"    {'TOTAL':<12s}: {len(labels):,}")
PYEOF

# ── 5. Retrain MLP ────────────────────────────────────────────────────────────
echo ""
echo "━━━ Step 4/4: Retrain MLP classifier ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Features : data/features.npy"
echo "  Labels   : data/labels.npy"
echo "  Output   : $MODEL_OUT"
echo ""

python scripts/train_model.py \
    --data   data/features.npy \
    --labels data/labels.npy \
    --mlp \
    --epochs 50 \
    --out    data/models

echo ""
echo "============================================================"
echo "  Retraining complete!"
echo "  New model : $MODEL_OUT"
[[ -f "$MODEL_BAK" ]] && echo "  Backup    : $MODEL_BAK"
echo "============================================================"
echo ""
echo "  Validate on the WWTP metagenome:"
echo "    plasflow2 run \\"
echo "      --input  data/test/GCA_054405655.1_ASM5440565v1_genomic.fna \\"
echo "      --output results/GCA_054405655_retrained/ \\"
echo "      --card-db  data/databases/card/card.dmnd \\"
echo "      --aro-index data/databases/card/aro_index.tsv \\"
echo "      --vfdb   data/databases/vfdb/vfdb_setA.dmnd \\"
echo "      --mge-db data/databases/mge/isfinder.dmnd \\"
echo "      --plasmid-threshold 0.95 \\"
echo "      --context wastewater \\"
echo "      --threads 10"
echo ""
echo "  Expected improvements after retrain:"
echo "    Unclassified : 124,698 → ideally <40,000"
echo "    Archaea      : 0 → properly identified (methanogens etc.)"
echo "    Plasmid FPR  : stays ~4-5% (threshold=0.95 unchanged)"
