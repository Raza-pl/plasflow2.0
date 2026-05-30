#!/usr/bin/env bash
# =============================================================================
# setup_and_assemble_srr10608981.sh
#
# Downloads SRR10608981, runs QC with fastp, assembles with MEGAHIT,
# and saves the final contigs ready for plasflow2 run.
#
# Usage:  bash scripts/setup_and_assemble_srr10608981.sh
#
# Requirements (installed automatically if conda is available):
#   fastp, megahit
#   wget or curl (for EBI download — faster than NCBI SRA tools)
# =============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRR="SRR10608981"
RAW_DIR="$PROJECT_DIR/data/raw/$SRR"
CLEAN_DIR="$PROJECT_DIR/data/clean/$SRR"
ASSEMBLY_DIR="$PROJECT_DIR/data/assembly/$SRR"
FINAL_FASTA="$PROJECT_DIR/data/test/${SRR}_contigs.fasta"

THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)

echo "============================================================"
echo "  PlasFlow v2 — Download + QC + Assembly pipeline"
echo "  Run: $SRR"
echo "  Threads: $THREADS"
echo "============================================================"

# ── 1. Check / install tools ─────────────────────────────────────────────────
install_if_missing() {
    local tool="$1"
    local pkg="${2:-$1}"
    if ! command -v "$tool" &>/dev/null; then
        echo "[setup] $tool not found — installing via conda..."
        if command -v conda &>/dev/null; then
            conda install -y -c bioconda -c conda-forge "$pkg" 2>&1 | tail -3
        elif command -v brew &>/dev/null; then
            brew install "$pkg"
        else
            echo "[ERROR] Neither conda nor brew found."
            echo "  Install conda:  https://docs.conda.io/en/latest/miniconda.html"
            echo "  Or brew:        https://brew.sh"
            exit 1
        fi
    else
        echo "[✓] $tool found: $(command -v $tool)"
    fi
}

install_if_missing fastp fastp
install_if_missing megahit megahit

# ── 2. Download reads ────────────────────────────────────────────────────────
mkdir -p "$RAW_DIR"

R1="$RAW_DIR/${SRR}_1.fastq.gz"
R2="$RAW_DIR/${SRR}_2.fastq.gz"

# EBI FTP path pattern: vol1/fastq/SRR{first6}/{last3}/{SRR}/{SRR}_{1,2}.fastq.gz
EBI_BASE="https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR106/081/${SRR}"

download_with_retry() {
    local url="$1"
    local dest="$2"
    local tmp="${dest}.tmp"

    if [[ -f "$dest" ]] && [[ $(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest" 2>/dev/null || echo 0) -gt 1000000 ]]; then
        echo "[✓] Already downloaded: $(basename "$dest") ($(du -sh "$dest" | cut -f1))"
        return 0
    fi

    echo "[download] $(basename "$dest") from EBI FTP (with resume + 5 retries)..."

    # wget: -c = resume, --tries = retry count, --timeout = socket timeout
    if command -v wget &>/dev/null; then
        wget -c --tries=5 --timeout=120 --retry-connrefused \
             --show-progress -q -O "$dest" "$url" && return 0
    fi

    # curl fallback: -C - = resume, --retry = retry count
    if command -v curl &>/dev/null; then
        for attempt in 1 2 3 4 5; do
            echo "  attempt $attempt/5..."
            curl -L -C - --progress-bar \
                 --connect-timeout 60 --max-time 3600 \
                 --retry 3 --retry-delay 10 \
                 -o "$dest" "$url" && return 0
            sleep 15
        done
    fi

    echo "[WARN] EBI FTP download failed — trying fasterq-dump as fallback..."
    return 1
}

# Try EBI FTP first
ebi_ok=true
download_with_retry "${EBI_BASE}/${SRR}_1.fastq.gz" "$R1" || ebi_ok=false
if $ebi_ok; then
    download_with_retry "${EBI_BASE}/${SRR}_2.fastq.gz" "$R2" || ebi_ok=false
fi

# Fallback: fasterq-dump (SRA toolkit)
if ! $ebi_ok || [[ ! -f "$R1" ]] || [[ ! -f "$R2" ]]; then
    echo "[fallback] Using fasterq-dump (SRA toolkit)..."
    if ! command -v fasterq-dump &>/dev/null; then
        echo "[setup] Installing sra-tools via conda..."
        conda install -y -c bioconda sra-tools 2>&1 | tail -3
    fi
    # fasterq-dump outputs uncompressed — we'll gzip after
    fasterq-dump "$SRR" \
        --outdir "$RAW_DIR" \
        --split-files \
        --threads "$THREADS" \
        --progress
    echo "[gzip] Compressing reads..."
    pigz -p "$THREADS" "$RAW_DIR/${SRR}_1.fastq" && mv "$RAW_DIR/${SRR}_1.fastq.gz" "$R1" 2>/dev/null || true
    pigz -p "$THREADS" "$RAW_DIR/${SRR}_2.fastq" && mv "$RAW_DIR/${SRR}_2.fastq.gz" "$R2" 2>/dev/null || true
    # pigz might not be installed — fallback to gzip
    [[ ! -f "$R1" ]] && gzip -c "$RAW_DIR/${SRR}_1.fastq" > "$R1" && rm "$RAW_DIR/${SRR}_1.fastq"
    [[ ! -f "$R2" ]] && gzip -c "$RAW_DIR/${SRR}_2.fastq" > "$R2" && rm "$RAW_DIR/${SRR}_2.fastq"
fi

# Verify downloads
for f in "$R1" "$R2"; do
    if [[ ! -f "$f" ]]; then
        echo "[ERROR] Missing file: $f"
        exit 1
    fi
    echo "[✓] $(basename "$f")  ($(du -sh "$f" | cut -f1))"
done

# ── 3. Quality control with fastp ────────────────────────────────────────────
mkdir -p "$CLEAN_DIR"

if [[ -f "$CLEAN_DIR/${SRR}_1.clean.fastq.gz" ]]; then
    echo "[✓] fastp output already exists — skipping QC"
else
    echo ""
    echo "[fastp] Running quality control..."
    fastp \
        --in1  "$R1" \
        --in2  "$R2" \
        --out1 "$CLEAN_DIR/${SRR}_1.clean.fastq.gz" \
        --out2 "$CLEAN_DIR/${SRR}_2.clean.fastq.gz" \
        --json "$CLEAN_DIR/fastp.json" \
        --html "$CLEAN_DIR/fastp.html" \
        --thread "$THREADS" \
        --detect_adapter_for_pe \
        --correction \
        --cut_front \
        --cut_tail \
        --cut_mean_quality 20 \
        --length_required 50 \
        --dedup \
        2>&1 | grep -E "Read|Filter|Filtering|Adapter|pass|fail|total" | head -20

    echo "[✓] QC complete — report: $CLEAN_DIR/fastp.html"
fi

# ── 4. Assemble with MEGAHIT ─────────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/data/assembly"

if [[ -d "$ASSEMBLY_DIR" ]]; then
    echo "[✓] MEGAHIT output already exists — skipping assembly"
else
    echo ""
    echo "[megahit] Assembling metagenome (this takes 20-60 min)..."
    megahit \
        -1 "$CLEAN_DIR/${SRR}_1.clean.fastq.gz" \
        -2 "$CLEAN_DIR/${SRR}_2.clean.fastq.gz" \
        -o "$ASSEMBLY_DIR" \
        --min-contig-len 500 \
        --k-list 21,29,39,59,79,99,119,141 \
        -t "$THREADS" \
        --verbose 2>&1 | grep -E "^\[|k =|contig" | tail -30
fi

# ── 5. Copy final contigs and print summary ───────────────────────────────────
mkdir -p "$(dirname "$FINAL_FASTA")"
MEGAHIT_CONTIGS="$ASSEMBLY_DIR/final.contigs.fa"

if [[ ! -f "$MEGAHIT_CONTIGS" ]]; then
    echo "[ERROR] MEGAHIT output not found: $MEGAHIT_CONTIGS"
    exit 1
fi

cp "$MEGAHIT_CONTIGS" "$FINAL_FASTA"

# Quick stats
echo ""
echo "============================================================"
echo "  Assembly complete!"
echo "============================================================"
python3 - "$FINAL_FASTA" <<'PYEOF'
import sys
from Bio import SeqIO
lengths = sorted([len(r) for r in SeqIO.parse(sys.argv[1], "fasta")], reverse=True)
total = sum(lengths)
n = len(lengths)
cumsum = 0
n50 = 0
for l in lengths:
    cumsum += l
    if cumsum >= total / 2:
        n50 = l
        break
over1k  = sum(1 for l in lengths if l >= 1000)
over10k = sum(1 for l in lengths if l >= 10000)
print(f"  Total contigs  : {n:,}")
print(f"  Total bp       : {total:,.0f} ({total/1e6:.1f} Mb)")
print(f"  Longest contig : {lengths[0]:,} bp")
print(f"  N50            : {n50:,} bp")
print(f"  Contigs ≥1kb   : {over1k:,}")
print(f"  Contigs ≥10kb  : {over10k:,}")
print(f"  FASTA saved to : {sys.argv[1]}")
PYEOF

echo ""
echo "  Next step — run PlasFlow v2:"
echo "    python -m plasflow2.cli run \\"
echo "      --input  $FINAL_FASTA \\"
echo "      --output results/${SRR}/ \\"
echo "      --model  data/models/mlp_v2.pt \\"
echo "      --card-db  data/databases/card/card.dmnd \\"
echo "      --aro-index data/databases/card/aro_index.tsv \\"
echo "      --context wastewater \\"
echo "      --threads $THREADS"
