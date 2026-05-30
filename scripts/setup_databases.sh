#!/usr/bin/env bash
# =============================================================================
# setup_databases.sh
#
# One-shot installer for all PlasFlow v2 annotation databases and tools.
#
# Downloads and builds:
#   - CARD  (already set up, just verified here)
#   - VFDB set A protein sequences → DIAMOND database
#   - ISfinder transposase sequences → DIAMOND database (MGE detection)
#
# Installs tools (via conda or brew):
#   - mob-suite  (plasmid mobility typing)
#   - diamond    (if not already present)
#
# Usage:  bash scripts/setup_databases.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DB_DIR="$PROJECT_DIR/data/databases"
THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }

echo "============================================================"
echo "  PlasFlow v2 — Database + Tool Setup"
echo "  DB root: $DB_DIR"
echo "  Threads: $THREADS"
echo "============================================================"
echo ""

# ── Helper: install a tool if missing ────────────────────────────────────────
install_tool() {
    local tool="$1"
    local conda_pkg="${2:-$1}"
    local brew_pkg="${3:-$1}"
    if command -v "$tool" &>/dev/null; then
        ok "$tool already installed: $(command -v $tool)"
        return 0
    fi
    warn "$tool not found — installing..."
    if command -v conda &>/dev/null; then
        conda install -y -c bioconda -c conda-forge "$conda_pkg" 2>&1 | tail -5
    elif command -v brew &>/dev/null; then
        brew install "$brew_pkg"
    else
        err "Neither conda nor brew found. Install one of:"
        err "  conda: https://docs.conda.io/en/latest/miniconda.html"
        err "  brew:  https://brew.sh"
        return 1
    fi
    if command -v "$tool" &>/dev/null; then
        ok "$tool installed"
    else
        err "Failed to install $tool"
        return 1
    fi
}

# ── Helper: download with progress ───────────────────────────────────────────
download() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        ok "Already downloaded: $(basename "$dest")"
        return 0
    fi
    echo "  Downloading $(basename "$dest")..."
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url"
    else
        curl -L --progress-bar -o "$dest" "$url"
    fi
}

# ── Helper: install mob-suite (conda often fails on ARM Mac; pip is reliable) ─
install_mob_suite() {
    if command -v mob_typer &>/dev/null; then
        ok "mob_typer already installed: $(command -v mob_typer)"
        return 0
    fi
    warn "mob_typer not found — installing mob-suite..."

    # Try conda first (works on x86 Linux/Mac)
    if command -v conda &>/dev/null; then
        echo "  Trying conda install..."
        conda install -y -c bioconda -c conda-forge mob-suite 2>&1 | tail -3 || true
    fi

    # Fallback: pip (works on ARM Mac and everywhere Python is available)
    if ! command -v mob_typer &>/dev/null; then
        echo "  conda failed or unavailable — trying pip install mob-suite..."
        pip install mob-suite --quiet 2>&1 | tail -3 || true
    fi

    if command -v mob_typer &>/dev/null; then
        ok "mob_typer installed: $(command -v mob_typer)"
    else
        err "mob-suite install failed — mobility typing will be skipped"
        warn "  Manual install:  pip install mob-suite"
        return 1
    fi
}

# ── 1. Tools ──────────────────────────────────────────────────────────────────
echo "─── Tools ───────────────────────────────────────────────────"
install_tool diamond diamond diamond
install_mob_suite || true   # non-fatal: pipeline skips mobility gracefully
echo ""

# ── 2. CARD (verify existing) ────────────────────────────────────────────────
echo "─── CARD (AMR) ──────────────────────────────────────────────"
CARD_DB="$DB_DIR/card/card.dmnd"
ARO_INDEX="$DB_DIR/card/aro_index.tsv"
if [[ -f "$CARD_DB" && -f "$ARO_INDEX" ]]; then
    ok "CARD database: $CARD_DB"
    ok "ARO index:     $ARO_INDEX"
else
    warn "CARD database not found at $DB_DIR/card/"
    warn "Run:  python -m plasflow2.cli setup"
    warn "or:   python -c \"from plasflow2.annotate.args import setup_card_db; setup_card_db('$DB_DIR/card')\""
fi
echo ""

# ── 3. VFDB set A ─────────────────────────────────────────────────────────────
echo "─── VFDB (Virulence Factors) ────────────────────────────────"
VFDB_DIR="$DB_DIR/vfdb"
VFDB_FASTA="$VFDB_DIR/VFDB_setA_pro.fas"
VFDB_DMND="$VFDB_DIR/vfdb_setA.dmnd"
mkdir -p "$VFDB_DIR"

if [[ -f "$VFDB_DMND" ]]; then
    ok "VFDB DIAMOND database already exists: $VFDB_DMND"
else
    # VFDB set A = experimentally validated VFs only (smaller, more specific)
    VFDB_URL="http://www.mgc.ac.cn/VFs/Down/VFDB_setA_pro.fas.gz"
    VFDB_GZ="$VFDB_DIR/VFDB_setA_pro.fas.gz"

    download "$VFDB_URL" "$VFDB_GZ"

    if [[ ! -f "$VFDB_FASTA" ]]; then
        echo "  Decompressing VFDB..."
        gunzip -k "$VFDB_GZ"
    fi

    echo "  Building DIAMOND database for VFDB..."
    diamond makedb \
        --in "$VFDB_FASTA" \
        --db "$VFDB_DIR/vfdb_setA" \
        --threads "$THREADS" \
        --quiet
    ok "VFDB database built: $VFDB_DMND"
fi
echo ""

# ── 4. MGE database (Pärnänen et al. 2018 — GitHub, no SSL issues) ───────────
echo "─── MGE database (IS elements + integrons + transposons) ────"
MGE_DIR="$DB_DIR/mge"
MGE_NT_FASTA="$MGE_DIR/MGEs_FINAL_99perc_trim.fasta"
MGE_AA_FASTA="$MGE_DIR/mge_proteins.faa"
MGE_DMND="$MGE_DIR/isfinder.dmnd"   # keep filename so --mge-db path stays valid
mkdir -p "$MGE_DIR"

if [[ -f "$MGE_DMND" ]]; then
    ok "MGE DIAMOND database already exists: $MGE_DMND"
else
    # Pärnänen et al. 2018 MGE database — direct GitHub download, no SSL issues.
    # Contains IS*, ISCR*, intI (integrons), tniA/B (Tn transposons) from NCBI.
    # ~2000 unique CDS sequences, 99% identity clustered.
    # Paper: Pärnänen et al. Nature Communications 2018;9:3891
    MGE_TGZ="$MGE_DIR/MGEs_FINAL_99perc_trim.fasta.tar.gz"
    MGE_URL="https://github.com/KatariinaParnanen/MobileGeneticElementDatabase/raw/master/MGEs_FINAL_99perc_trim.fasta.tar.gz"

    download "$MGE_URL" "$MGE_TGZ"

    if [[ ! -f "$MGE_NT_FASTA" ]]; then
        echo "  Extracting archive..."
        tar -xzf "$MGE_TGZ" -C "$MGE_DIR"
        # Rename if extracted with slightly different name
        find "$MGE_DIR" -name "MGEs_FINAL*.fasta" ! -name "mge_proteins.faa" \
             -exec mv {} "$MGE_NT_FASTA" \; 2>/dev/null || true
    fi

    if [[ ! -f "$MGE_NT_FASTA" ]]; then
        err "MGE FASTA extraction failed — check $MGE_TGZ"
        exit 1
    fi

    NT_COUNT=$(grep -c "^>" "$MGE_NT_FASTA")
    echo "  Nucleotide CDS sequences: $NT_COUNT"

    # Translate CDS → protein with biopython, then build DIAMOND database
    echo "  Translating CDS to protein..."
    python3 - "$MGE_NT_FASTA" "$MGE_AA_FASTA" <<'PYEOF'
import sys
from Bio import SeqIO
from Bio.Seq import Seq

in_fa, out_fa = sys.argv[1], sys.argv[2]
written = 0
with open(out_fa, "w") as fh:
    for rec in SeqIO.parse(in_fa, "fasta"):
        nt = str(rec.seq).upper().replace("-", "N")
        # Pad to multiple of 3
        if len(nt) % 3:
            nt += "N" * (3 - len(nt) % 3)
        aa = str(Seq(nt).translate(to_stop=True))
        if len(aa) >= 30:          # skip very short/truncated ORFs
            fh.write(f">{rec.id} {rec.description[len(rec.id):].strip()}\n{aa}\n")
            written += 1
print(f"  Translated {written} proteins → {out_fa}")
PYEOF

    AA_COUNT=$(grep -c "^>" "$MGE_AA_FASTA")
    echo "  Protein sequences: $AA_COUNT"

    echo "  Building DIAMOND database..."
    diamond makedb \
        --in "$MGE_AA_FASTA" \
        --db "$MGE_DIR/isfinder" \
        --threads "$THREADS" \
        --quiet
    ok "MGE database built: $MGE_DMND  ($AA_COUNT proteins, from $NT_COUNT CDS)"
fi
echo ""

# ── 5. mob-suite database (if mob_typer is installed) ────────────────────────
echo "─── MOB-suite databases ─────────────────────────────────────"
if command -v mob_typer &>/dev/null; then
    if [[ -d "$HOME/.mob_suite" || -d "/usr/local/share/mob-suite" ]]; then
        ok "mob-suite databases already initialised"
    else
        echo "  Initialising mob-suite databases (downloads ~500 MB)..."
        mob_init 2>&1 | tail -5 && ok "mob-suite databases ready" || \
            warn "mob_init failed — run 'mob_init' manually after installation"
    fi
else
    warn "mob_typer not installed — mobility typing will be skipped"
    warn "Install: conda install -c bioconda mob-suite"
fi
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Setup complete! Database locations:"
echo ""
printf "  %-20s %s\n" "CARD (ARG):"  "${CARD_DB:-NOT FOUND}"
printf "  %-20s %s\n" "VFDB:"        "${VFDB_DMND:-NOT FOUND}"
printf "  %-20s %s\n" "MGE database:" "${MGE_DMND:-NOT FOUND}"
printf "  %-20s %s\n" "mob_typer:"   "$(command -v mob_typer 2>/dev/null || echo 'NOT INSTALLED')"
echo ""
echo "  Full pipeline run example:"
echo "    python -m plasflow2.cli run \\"
echo "      --input  contigs.fasta \\"
echo "      --output results/my_run/ \\"
echo "      --model  data/models/mlp_v2.pt \\"
echo "      --card-db   $CARD_DB \\"
echo "      --aro-index $ARO_INDEX \\"
echo "      --vfdb      $VFDB_DMND \\"
echo "      --mge-db    $MGE_DMND \\"
echo "      --context wastewater \\"
echo "      --threads $THREADS"
echo "============================================================"
