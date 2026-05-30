#!/usr/bin/env bash
# =============================================================================
# download_and_assemble.sh
#
# Downloads any SRA run, runs QC with fastp, assembles with MEGAHIT.
# Uses prefetch + fasterq-dump (most reliable; built-in resume support).
#
# Usage:
#   bash scripts/download_and_assemble.sh SRR29792380
#   bash scripts/download_and_assemble.sh SRR10608981
#
# The SRR accession is the only required argument.
# =============================================================================

set -euo pipefail

SRR="${1:?Usage: $0 <SRR_accession>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RAW_DIR="$PROJECT_DIR/data/raw/$SRR"
CLEAN_DIR="$PROJECT_DIR/data/clean/$SRR"
ASSEMBLY_DIR="$PROJECT_DIR/data/assembly/$SRR"
FINAL_FASTA="$PROJECT_DIR/data/test/${SRR}_contigs.fasta"

THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)

echo "============================================================"
echo "  PlasFlow v2 — Download + QC + Assembly"
echo "  Run      : $SRR"
echo "  Threads  : $THREADS"
echo "  Output   : $FINAL_FASTA"
echo "============================================================"

# ── helpers ──────────────────────────────────────────────────────────────────
install_if_missing() {
    local tool="$1" pkg="${2:-$1}"
    if ! command -v "$tool" &>/dev/null; then
        echo "[setup] $tool not found — installing via conda..."
        conda install -y -c bioconda -c conda-forge "$pkg" 2>&1 | tail -3
    else
        echo "[✓] $tool: $(command -v "$tool")"
    fi
}

# ── 1. Install tools ──────────────────────────────────────────────────────────
install_if_missing fastp
install_if_missing megahit
install_if_missing prefetch sra-tools
install_if_missing fasterq-dump sra-tools

# ── 2. Download reads ────────────────────────────────────────────────────────
mkdir -p "$RAW_DIR"

R1="$RAW_DIR/${SRR}_1.fastq.gz"
R2="$RAW_DIR/${SRR}_2.fastq.gz"

file_ok() {
    local f="$1" min_bytes="${2:-1000000}"
    [[ -f "$f" ]] && \
        [[ $(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0) -gt $min_bytes ]]
}

if file_ok "$R1" && file_ok "$R2"; then
    echo "[✓] Reads already downloaded:"
    echo "    R1: $(du -sh "$R1" | cut -f1)  $R1"
    echo "    R2: $(du -sh "$R2" | cut -f1)  $R2"
else
    # -- Try EBI FTP first (fast; no account needed) --------------------------
    # Dynamically resolve EBI FTP URLs via the ENA portal API
    echo "[download] Resolving EBI FTP URLs for $SRR..."
    EBI_API="https://www.ebi.ac.uk/ena/portal/api/filereport?accession=${SRR}&result=read_run&fields=fastq_ftp"
    EBI_URLS=$(curl -sf --connect-timeout 15 "$EBI_API" 2>/dev/null | \
               tail -1 | tr ';' '\n' | grep fastq | head -2 || true)

    ebi_ok=false
    if [[ -n "$EBI_URLS" ]]; then
        echo "[download] EBI FTP paths found — downloading with wget (resume enabled)..."
        i=1
        for url in $EBI_URLS; do
            dest="${RAW_DIR}/${SRR}_${i}.fastq.gz"
            echo "  [R${i}] $url"
            wget -c --tries=5 --timeout=180 --retry-connrefused \
                 --show-progress -q -O "$dest" \
                 "https://${url#ftp://}" && i=$((i+1)) || true
        done
        file_ok "$R1" && file_ok "$R2" && ebi_ok=true
    fi

    # -- Fallback: prefetch + fasterq-dump (NCBI SRA) -------------------------
    if ! $ebi_ok; then
        echo "[download] EBI unavailable — using NCBI prefetch + fasterq-dump..."
        SRA_CACHE="$RAW_DIR/sra_cache"
        mkdir -p "$SRA_CACHE"

        # prefetch: downloads .sra file with built-in resume support
        prefetch "$SRR" \
            --output-directory "$SRA_CACHE" \
            --max-size 30G \
            --resume yes \
            --progress

        SRA_FILE="$SRA_CACHE/$SRR/$SRR.sra"
        [[ ! -f "$SRA_FILE" ]] && SRA_FILE="$(find "$SRA_CACHE" -name "*.sra" | head -1)"

        echo "[convert] fasterq-dump → FASTQ (split-files, $THREADS threads)..."
        fasterq-dump "$SRA_FILE" \
            --outdir "$RAW_DIR" \
            --split-files \
            --threads "$THREADS" \
            --progress

        echo "[gzip] Compressing reads..."
        if command -v pigz &>/dev/null; then
            pigz -p "$THREADS" "$RAW_DIR/${SRR}_1.fastq"
            pigz -p "$THREADS" "$RAW_DIR/${SRR}_2.fastq"
        else
            gzip "$RAW_DIR/${SRR}_1.fastq"
            gzip "$RAW_DIR/${SRR}_2.fastq"
        fi
    fi
fi

# Final check
for f in "$R1" "$R2"; do
    if ! file_ok "$f"; then
        echo "[ERROR] Missing or empty: $f"
        exit 1
    fi
    echo "[✓] $(basename "$f")  ($(du -sh "$f" | cut -f1))"
done

# ── 3. QC with fastp ─────────────────────────────────────────────────────────
mkdir -p "$CLEAN_DIR"
C1="$CLEAN_DIR/${SRR}_1.clean.fastq.gz"
C2="$CLEAN_DIR/${SRR}_2.clean.fastq.gz"

if file_ok "$C1" && file_ok "$C2"; then
    echo "[✓] fastp output already exists — skipping QC"
else
    echo ""
    echo "[fastp] Quality control..."
    fastp \
        --in1  "$R1" \
        --in2  "$R2" \
        --out1 "$C1" \
        --out2 "$C2" \
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
        2>&1 | grep -E "reads passed|reads failed|Filtering|total reads" | head -10

    echo "[✓] QC done — HTML report: $CLEAN_DIR/fastp.html"
fi

# ── 4. Assembly with MEGAHIT ─────────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/data/assembly"

if [[ -d "$ASSEMBLY_DIR" ]]; then
    echo "[✓] MEGAHIT output already exists — skipping assembly"
else
    echo ""
    echo "[megahit] Assembling metagenome (20–60 min on $THREADS threads)..."
    megahit \
        -1 "$C1" \
        -2 "$C2" \
        -o "$ASSEMBLY_DIR" \
        --min-contig-len 500 \
        --k-list 21,29,39,59,79,99,119,141 \
        -t "$THREADS" \
        2>&1 | grep -E "^\[|k =|ALL DONE|contigs" | tail -20
fi

# ── 5. Copy contigs + print stats ────────────────────────────────────────────
mkdir -p "$(dirname "$FINAL_FASTA")"
MEGAHIT_OUT="$ASSEMBLY_DIR/final.contigs.fa"

if [[ ! -f "$MEGAHIT_OUT" ]]; then
    echo "[ERROR] MEGAHIT output not found: $MEGAHIT_OUT"
    exit 1
fi

cp "$MEGAHIT_OUT" "$FINAL_FASTA"

echo ""
echo "============================================================"
echo "  Assembly complete!"
echo "============================================================"
python3 - "$FINAL_FASTA" <<'PYEOF'
import sys
try:
    from Bio import SeqIO
    lengths = sorted([len(r) for r in SeqIO.parse(sys.argv[1], "fasta")], reverse=True)
except ImportError:
    # fallback: count '>' lines and rough length from file size
    lengths = []
    with open(sys.argv[1]) as f:
        cur = 0
        for line in f:
            if line.startswith('>'):
                if cur: lengths.append(cur)
                cur = 0
            else:
                cur += len(line.strip())
        if cur: lengths.append(cur)
    lengths.sort(reverse=True)

total  = sum(lengths)
n      = len(lengths)
cumsum = 0
n50    = 0
for l in lengths:
    cumsum += l
    if cumsum >= total / 2:
        n50 = l
        break

print(f"  Total contigs  : {n:,}")
print(f"  Total bases    : {total:,} ({total/1e6:.1f} Mb)")
print(f"  Longest contig : {lengths[0]:,} bp")
print(f"  N50            : {n50:,} bp")
print(f"  Contigs ≥ 1 kb : {sum(1 for l in lengths if l >= 1000):,}")
print(f"  Contigs ≥10 kb : {sum(1 for l in lengths if l >= 10000):,}")
PYEOF

echo ""
echo "  FASTA: $FINAL_FASTA"
echo ""
echo "  Run PlasFlow v2:"
echo "    python -m plasflow2.cli run \\"
echo "      --input  $FINAL_FASTA \\"
echo "      --output results/${SRR}/ \\"
echo "      --model  data/models/mlp_v2.pt \\"
echo "      --card-db  data/databases/card/card.dmnd \\"
echo "      --aro-index data/databases/card/aro_index.tsv \\"
echo "      --plasmid-threshold 0.95 \\"
echo "      --context wastewater \\"
echo "      --threads $THREADS"
