#!/usr/bin/env bash
# =============================================================================
# retrain_with_more_chromosomes.sh
#
# Downloads diverse RefSeq chromosomes, rebuilds the training dataset, and
# retrains the MLP classifier. Run this once to fix the chromosome class.
#
# Problem being fixed
# -------------------
# The current model was trained on chromosome windows from only 40 source
# genomes, massively oversampled to 95,000 windows. The model memorises those
# 40 genomes' k-mer profiles instead of learning general chromosome features.
# Result: novel chromosomal contigs in metagenomes are "unclassified" or
# wrongly called plasmid/phage.
#
# Fix: download 1,000 diverse RefSeq genomes (14 phyla), tile into windows,
# rebuild the full dataset, and retrain.
#
# Runtime estimate (no API key):
#   Download  : ~1–2 h  (1,000 genomes × ~1.5 MB compressed)
#   Build     : ~20 min (feature extraction on 285k windows)
#   Retrain   : ~15 min (50 epochs, early stopping)
#   Total     : ~2–3 h
#
# Usage:
#   bash scripts/retrain_with_more_chromosomes.sh
#   bash scripts/retrain_with_more_chromosomes.sh --count 500   # faster, less diversity
#   bash scripts/retrain_with_more_chromosomes.sh --count 2000  # better, slower
#   bash scripts/retrain_with_more_chromosomes.sh --api-key YOUR_NCBI_KEY  # 3× faster
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CHROM_DIR="$PROJECT_DIR/data/chromosomes"
DATA_DIR="$PROJECT_DIR/data"
MODEL_OUT="$DATA_DIR/models/mlp_v2.pt"
MODEL_BAK="$DATA_DIR/models/mlp_v2_backup_$(date +%Y%m%d_%H%M%S).pt"

GENOME_COUNT=1000
API_KEY=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --count)    GENOME_COUNT="$2"; shift 2 ;;
        --api-key)  API_KEY="$2";      shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  PlasFlow v2 — Chromosome retraining pipeline"
echo "  Genomes to download : $GENOME_COUNT"
echo "  Chromosome dir      : $CHROM_DIR"
echo "  Model output        : $MODEL_OUT"
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

# ── 2. Download diverse RefSeq chromosomes ────────────────────────────────────
echo ""
echo "━━━ Step 1/3: Download chromosomes ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Target: $GENOME_COUNT genomes across 14 bacterial phyla"
echo "  Output: $CHROM_DIR/"
echo ""

ALREADY=$(find "$CHROM_DIR" -name "*.fna" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$ALREADY" -ge "$GENOME_COUNT" ]]; then
    echo "[✓] Already have $ALREADY genomes in $CHROM_DIR — skipping download"
else
    echo "[info] Found $ALREADY existing genomes — downloading remaining..."
    API_FLAG=""
    [[ -n "$API_KEY" ]] && API_FLAG="--api-key $API_KEY"

    python scripts/download_refseq_chromosomes.py \
        --count "$GENOME_COUNT" \
        --outdir "$CHROM_DIR" \
        $API_FLAG \
        --email "plasflow2@example.com"

    DOWNLOADED=$(find "$CHROM_DIR" -name "*.fna" | wc -l | tr -d ' ')
    echo ""
    echo "[✓] Chromosomes ready: $DOWNLOADED .fna files in $CHROM_DIR"
fi

# ── 3. Rebuild training dataset ───────────────────────────────────────────────
echo ""
echo "━━━ Step 2/3: Rebuild training dataset ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Sources:"
echo "    Plasmid  — data/databases/plasmids/ (PLSDB + RefSeq + COMPASS)"
echo "    Chrom    — $CHROM_DIR/ ($GENOME_COUNT genomes, tiled to windows)"
echo "    Phage    — data/databases/inphared/ (INPHARED)"
echo ""

python scripts/build_dataset.py \
    --plasmid-dir  data/databases/plasmids/ \
    --chrom-dir    "$CHROM_DIR" \
    --data-dir     data/databases/ \
    --max-per-class 95000 \
    --out          data/

echo ""
python3 - <<'PYEOF'
import numpy as np, collections
labels = np.load("data/labels.npy")
counts = dict(sorted(collections.Counter(labels.tolist()).items()))
names = {0: "plasmid", 1: "chromosome", 2: "phage", 3: "archaea"}
print("  New dataset label distribution:")
for k, v in counts.items():
    print(f"    {names.get(k, str(k)):<12s}: {v:,}")
print(f"    {'TOTAL':<12s}: {len(labels):,}")
PYEOF

# ── 4. Retrain MLP ────────────────────────────────────────────────────────────
echo ""
echo "━━━ Step 3/3: Retrain MLP classifier ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Features : data/features.npy"
echo "  Labels   : data/labels.npy"
echo "  Output   : $MODEL_OUT"
echo ""

python scripts/train_model.py \
    --data   data/features.npy \
    --labels data/labels.npy \
    --mlp \
    --epochs 50 \
    --output "$MODEL_OUT"

echo ""
echo "============================================================"
echo "  Retraining complete!"
echo "  New model : $MODEL_OUT"
[[ -f "$MODEL_BAK" ]] && echo "  Backup    : $MODEL_BAK"
echo "============================================================"
echo ""
echo "  Validate on the WWTP metagenome:"
echo "    python -m plasflow2.cli run \\"
echo "      --input  data/test/GCA_054405655.1_ASM5440565v1_genomic.fna \\"
echo "      --output results/GCA_054405655_retrained/ \\"
echo "      --model  $MODEL_OUT \\"
echo "      --card-db  data/databases/card/card.dmnd \\"
echo "      --aro-index data/databases/card/aro_index.tsv \\"
echo "      --plasmid-threshold 0.95 \\"
echo "      --context wastewater \\"
echo "      --threads 10"
echo ""
echo "  Expected improvement:"
echo "    Unclassified: 124,698 → ideally <50,000 (novel chromosomes now"
echo "                  recognised instead of falling through)"
echo "    Plasmid FPR : stays ~4–5% (threshold=0.95 unchanged)"
