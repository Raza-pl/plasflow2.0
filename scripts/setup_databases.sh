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

# ── Helper: download with progress (optional extra flags forwarded to curl) ───
download() {
    local url="$1"
    local dest="$2"
    shift 2
    local extra_flags=("$@")   # e.g. --insecure
    if [[ -f "$dest" ]]; then
        ok "Already downloaded: $(basename "$dest")"
        return 0
    fi
    echo "  Downloading $(basename "$dest")..."
    if command -v wget &>/dev/null; then
        # map --insecure → --no-check-certificate for wget
        local wget_flags=()
        for f in "${extra_flags[@]}"; do
            [[ "$f" == "--insecure" ]] && wget_flags+=("--no-check-certificate") || wget_flags+=("$f")
        done
        wget -q --show-progress "${wget_flags[@]}" -O "$dest" "$url"
    else
        curl -L --progress-bar "${extra_flags[@]}" -o "$dest" "$url"
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

# ── 4. ISfinder (MGE / IS elements) ──────────────────────────────────────────
echo "─── ISfinder (MGE / IS elements) ────────────────────────────"
MGE_DIR="$DB_DIR/mge"
ISFINDER_FASTA="$MGE_DIR/ISfinder-sequences.fasta"
ISFINDER_DMND="$MGE_DIR/isfinder.dmnd"
mkdir -p "$MGE_DIR"

if [[ -f "$ISFINDER_DMND" ]]; then
    ok "ISfinder DIAMOND database already exists: $ISFINDER_DMND"
else
    # ISfinder has a broken SSL cert — use --insecure to bypass it.
    # The file content is legitimate; the cert is just self-signed/expired.
    isfinder_ok=false

    # Source 1: ISfinder official (broken SSL — use --insecure)
    echo "  Trying ISfinder official site (--insecure for SSL bypass)..."
    download "https://isfinder.biotoul.fr/download/ISfinder-sequences.fasta" \
             "$ISFINDER_FASTA" --insecure && \
        grep -q "^>" "$ISFINDER_FASTA" 2>/dev/null && isfinder_ok=true || true

    # Source 2: Zenodo archive of ISfinder v21 proteins
    if ! $isfinder_ok; then
        rm -f "$ISFINDER_FASTA"
        warn "Official site failed — trying Zenodo archive..."
        download "https://zenodo.org/record/7556006/files/ISfinder-sequences.fasta" \
                 "$ISFINDER_FASTA" && \
            grep -q "^>" "$ISFINDER_FASTA" 2>/dev/null && isfinder_ok=true || true
    fi

    # Source 3: ISfinder sequences via NCBI Entrez (small Python fallback)
    if ! $isfinder_ok; then
        rm -f "$ISFINDER_FASTA"
        warn "Zenodo failed — fetching IS element proteins from NCBI..."
        python3 - "$ISFINDER_FASTA" <<'PYEOF'
import sys, time, urllib.request, json
out = sys.argv[1]
# Search NCBI protein for ISfinder transposases
# Query: transposase[Title] AND "IS element"[Title] with reasonable size
base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
search_url = (f"{base}/esearch.fcgi?db=protein&term=transposase[Title]+"
              "AND+insertion+sequence[Title]&retmax=5000&retmode=json&usehistory=y")
with urllib.request.urlopen(search_url) as r:
    data = json.loads(r.read())
webenv = data["esearchresult"]["webenv"]
query_key = data["esearchresult"]["querykey"]
count = int(data["esearchresult"]["count"])
print(f"  Found {count} transposase entries on NCBI")
written = 0
batch = 500
with open(out, "w") as fh:
    for start in range(0, min(count, 5000), batch):
        fetch_url = (f"{base}/efetch.fcgi?db=protein&query_key={query_key}"
                     f"&WebEnv={webenv}&retstart={start}&retmax={batch}"
                     f"&rettype=fasta&retmode=text")
        with urllib.request.urlopen(fetch_url) as r:
            chunk = r.read().decode()
        fh.write(chunk)
        written += chunk.count(">")
        print(f"  Fetched {written} sequences...", end="\r")
        time.sleep(0.35)
print(f"\n  Wrote {written} transposase sequences to {out}")
PYEOF
        grep -q "^>" "$ISFINDER_FASTA" 2>/dev/null && isfinder_ok=true || true
    fi

    if $isfinder_ok; then
        echo "  Validating FASTA..."
        NSEQS=$(grep -c "^>" "$ISFINDER_FASTA")
        echo "  Sequences: $NSEQS"
        echo "  Building DIAMOND database for ISfinder..."
        diamond makedb \
            --in "$ISFINDER_FASTA" \
            --db "$MGE_DIR/isfinder" \
            --threads "$THREADS" \
            --quiet
        ok "ISfinder database built: $ISFINDER_DMND  ($NSEQS sequences)"
    else
        err "All ISfinder download sources failed."
        warn "Manual fix:"
        warn "  1. Open https://isfinder.biotoul.fr/download.php in your browser"
        warn "  2. Download the protein FASTA file"
        warn "  3. cp ~/Downloads/ISfinder-sequences.fasta $ISFINDER_FASTA"
        warn "  4. diamond makedb --in $ISFINDER_FASTA --db $MGE_DIR/isfinder --threads $THREADS"
    fi
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
printf "  %-20s %s\n" "CARD (ARG):"     "${CARD_DB:-NOT FOUND}"
printf "  %-20s %s\n" "VFDB:"           "${VFDB_DMND:-NOT FOUND}"
printf "  %-20s %s\n" "ISfinder (MGE):" "${ISFINDER_DMND:-NOT FOUND}"
printf "  %-20s %s\n" "mob_typer:"      "$(command -v mob_typer 2>/dev/null || echo 'NOT INSTALLED')"
echo ""
echo "  Full pipeline run example:"
echo "    python -m plasflow2.cli run \\"
echo "      --input  contigs.fasta \\"
echo "      --output results/my_run/ \\"
echo "      --model  data/models/mlp_v2.pt \\"
echo "      --card-db   $CARD_DB \\"
echo "      --aro-index $ARO_INDEX \\"
echo "      --vfdb      $VFDB_DMND \\"
echo "      --mge-db    $ISFINDER_DMND \\"
echo "      --context wastewater \\"
echo "      --threads $THREADS"
echo "============================================================"
