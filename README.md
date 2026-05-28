# PlasFlow v2

[![CI](https://github.com/Raza-pl/plasflow2.0/actions/workflows/ci.yml/badge.svg)](https://github.com/Raza-pl/plasflow2.0/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Tests](https://img.shields.io/badge/tests-204%20passing-brightgreen.svg)](#development)

**PlasFlow v2** classifies metagenomic contigs as plasmid, chromosome, phage, or archaea, annotates antibiotic resistance genes (ARGs) via CARD, determines mobility class with MOB-suite, assigns taxonomy via DIAMOND + GTDB LCA, scores AMR risk (0–10), and produces an interactive HTML report — all in one command.

This is a full rewrite of [PlasFlow v1](https://github.com/smaegol/PlasFlow) (Krawczyk et al., *Nucleic Acids Research* 2018) on a modern stack, tested on real wastewater metagenomes with 170,000+ contigs.

---

## What's new in v2

| Feature | v1 | v2 |
|---|---|---|
| Python version | 3.5 / TensorFlow 0.10 | 3.10+ / PyTorch 2.x |
| Classes | plasmid vs chromosome | plasmid · chromosome · **phage** · **archaea** |
| Architecture | TF neural net | **4-class MLP (97.4% accuracy)** + Random Forest |
| ARG annotation | ✗ | DIAMOND + CARD + **SARG** (dual-DB) |
| Mobility typing | ✗ | MOB-suite (conjugative / mobilizable / non-mobilizable) |
| Taxonomy | ✗ | **DIAMOND + GTDB LCA (Kaiju-style)** per contig |
| AMR risk score | ✗ | 0–10 score with evidence breakdown |
| Output | TSV only | TSV + FASTA bins + **interactive HTML report** |
| Install | conda-only | pip / Poetry + `plasflow2 setup` guide |
| Apple Silicon | ✗ | MPS (M1/M2/M3) accelerated |
| Test suite | ✗ | **175 tests** (unit + integration) |

---

## Real-world performance

Tested on a real wastewater metagenome assembly (GCA_054405655, 408 MB, 170,107 contigs):

| Class | Contigs | % | Mean confidence |
|---|---|---|---|
| Chromosome | 151,119 | 88.8% | 0.992 |
| Plasmid | 6,860 | 4.0% | 0.910 |
| Phage | 5,883 | 3.5% | 0.937 |
| Unclassified | 6,245 | 3.7% | 0.588 |

Classified in ~7 minutes on Apple M1 (MPS). Unclassified contigs are those where no class exceeded the 0.70 confidence threshold.

> **Note:** The classifier is designed for WGS assembly contigs (1 kb – 500 kb). Feeding complete reference chromosomes (>1 Mb) as single sequences is not a supported use case.

---

## Installation

### Docker (zero-setup)

```bash
# Build the image
docker build -t plasflow2 .

# Classify only (no databases required)
docker run --rm \
  -v /path/to/input:/data/input:ro \
  -v /path/to/results:/results \
  plasflow2 classify \
    --input  /data/input/assembly.fasta \
    --output /results/predictions.tsv

# Full pipeline with CARD + GTDB databases mounted
docker run --rm \
  -v /path/to/databases:/data/databases:ro \
  -v /path/to/input:/data/input:ro \
  -v /path/to/results:/results \
  plasflow2 run \
    --input   /data/input/assembly.fasta \
    --output  /results/ \
    --card-db /data/databases/card/card.dmnd \
    --aro-index /data/databases/card/aro_index.tsv \
    --threads 8

# Or with docker-compose (edit PLASFLOW_* env vars to set paths)
docker compose run plasflow2
```

### From source (recommended during alpha)

```bash
git clone https://github.com/Raza-pl/plasflow2.0
cd plasflow2.0
pip install poetry
poetry install
```

### Apple Silicon (M1/M2/M3)

PyTorch automatically uses the MPS backend on Apple Silicon — no extra configuration needed.

```bash
git clone https://github.com/Raza-pl/plasflow2.0
cd plasflow2.0
pip install poetry && poetry install
```

### External dependencies

The following tools must be available on your `PATH`:

| Tool | Purpose | Install |
|---|---|---|
| [DIAMOND](https://github.com/bbuchfink/diamond) | ARG annotation | `conda install -c bioconda diamond` |
| [MOB-suite](https://github.com/phac-nml/mob-suite) | Mobility typing | `conda install -c conda-forge -c bioconda mob_suite` |
| [Prodigal](https://github.com/hyattpd/Prodigal) | ORF prediction (via pyrodigal) | bundled via `pip install pyrodigal` |

---

## Model training

Pre-trained weights are in `data/models/mlp_v2.pt` (trained on 30,000 balanced fragments from PLSDB plasmids + RefSeq chromosomes + INPHARED phages).

To retrain from scratch:

```bash
# 1. Build the training dataset
python scripts/build_dataset.py \
  --plasmid-dir  data/plasmids/ \
  --chrom-dir    data/chromosomes/ \
  --phage-dir    data/phages/ \
  --output       data/features.npy \
  --labels       data/labels.npy \
  --n-per-class  7500

# 2. Train the MLP
python scripts/train_model.py \
  --mlp \
  --data   data/features.npy \
  --labels data/labels.npy \
  --epochs 50 \
  --output data/models/mlp_v2.pt
```

---

## Quickstart

### Full pipeline

```bash
plasflow2 run \
  --input        assembly.fasta \
  --output       ./results/ \
  --model        data/models/mlp_v2.pt \
  --card-db      data/databases/card/card.dmnd \
  --aro-index    data/databases/card/aro_index.tsv \
  --taxonomy-db  data/databases/gtdb/gtdb_r220.dmnd \
  --taxon-map    data/databases/gtdb/taxon_map.tsv \
  --context      wastewater \
  --threads      8
```

Skip optional modules when databases are unavailable:

```bash
plasflow2 run --input assembly.fasta --output ./results/ \
  --skip-mobility --skip-taxonomy
```

**Outputs in `./results/`:**

| File | Description |
|---|---|
| `predictions.tsv` | Per-contig classification, confidence, and per-class scores |
| `plasmid.fasta` | Classified plasmid sequences |
| `chromosome.fasta` | Classified chromosomal sequences |
| `phage.fasta` | Classified phage sequences |
| `archaea.fasta` | Classified archaeal sequences |
| `unclassified.fasta` | Low-confidence sequences (below threshold) |
| `annotations.json` | ARG hits, mobility type, taxonomy, risk score per plasmid contig |
| `report.html` | Self-contained interactive HTML report (Plotly + DataTables) |

### Classify only (no DIAMOND or MOB-suite required)

```bash
plasflow2 classify \
  --input  assembly.fasta \
  --output predictions.tsv \
  --model  data/models/mlp_v2.pt
```

### Annotate plasmids only

```bash
plasflow2 annotate \
  --input     plasmids.fasta \
  --output    annotations/ \
  --card-db   data/databases/card/card.dmnd \
  --aro-index data/databases/card/aro_index.tsv \
  --threads   8
```

### Regenerate report from existing outputs

```bash
plasflow2 report \
  --annotations results/annotations.json \
  --predictions results/predictions.tsv \
  --output      results/report.html \
  --context     wastewater
```

---

## CLI reference

```
plasflow2 [--verbose] COMMAND [OPTIONS]

Commands:
  run        Full pipeline: classify → annotate → taxonomy → risk → report
  classify   Classify sequences only; write predictions.tsv
  annotate   Annotate plasmid sequences with ARGs and mobility
  report     Build HTML report from existing annotations + predictions
  setup      Print installation guide for all external dependencies

Options for plasflow2 run:
  --input / -i        Input assembly FASTA (required)
  --output / -o       Output directory (required)
  --model             Path to .pt model weights
  --card-db           CARD DIAMOND database (.dmnd)
  --aro-index         CARD ARO index (aro_index.tsv)
  --sarg-db           SARG DIAMOND database (.dmnd) for dual-DB ARG annotation
  --taxonomy-db       DIAMOND database built from GTDB-r220 / RefSeq proteins
  --taxon-map         2-column accession→lineage TSV (improves LCA accuracy)
  --threshold         Confidence threshold, default 0.7
  --context           clinical | wastewater | environmental | unspecified
  --threads           CPU threads for DIAMOND/MOB-suite, default 8
  --min-length        Minimum contig length in bp, default 1000
  --skip-mobility     Skip MOB-suite (use when mob_typer is unavailable)
  --skip-taxonomy     Skip taxonomy annotation (use when no GTDB DB available)
  --verbose / -v      Enable debug logging
```

Run `plasflow2 setup` for step-by-step instructions on downloading and indexing all databases.

---

## Taxonomy annotation

Each contig is annotated with its lowest common ancestor (LCA) taxon using DIAMOND blastx against the GTDB-r220 representative protein database. The algorithm is Kaiju-style: it collects the top-10 hits per contig, walks from domain → species, and accepts the deepest rank where a strict majority (>50%) of hits agree. Ties at a rank are resolved upward to the parent — so a 50/50 Escherichia/Klebsiella split correctly lands at family (Enterobacteriaceae) rather than arbitrarily picking one genus.

| Parameter | Default | Description |
|---|---|---|
| `--taxonomy-db` | — | DIAMOND .dmnd built from GTDB-r220 proteins |
| `--taxon-map` | — | 2-column accession→lineage TSV (output of `build_gtdb_taxon_map`) |
| `--skip-taxonomy` | False | Skip the step entirely |

Taxonomy is stored per-contig in `annotations.json` (`lineage`, `rank`, `taxon`, `agreement`) and shown as the "Taxonomy (LCA)" column in the HTML report.

---

## ARG annotation (CARD + SARG)

PlasFlow v2 annotates antibiotic resistance genes using DIAMOND BLASTp against one or both databases:

**CARD** (Comprehensive Antibiotic Resistance Database) is the default — strict thresholds (90% identity, 80% coverage), rich metadata (ARO accession, AMR family, resistance mechanism).

**SARG** (Structured ARG database) is optional (`--sarg-db`) — looser thresholds (80% identity, 80% coverage), captures more divergent homologues not represented in CARD. When both are enabled, CARD hits take precedence per ORF and SARG contributes supplementary hits for genes only found in SARG.

The DB source (CARD / SARG) is shown as a colour-coded badge in the HTML report and serialised as a `source` field in `annotations.json`.

**SARG setup (one-time):**
```bash
# Download SARG FASTA from https://smile.hku.hk/SARGs
mkdir -p data/databases/sarg
diamond makedb --in sarg.fasta -d data/databases/sarg/sarg
```

**Run with dual-DB annotation:**
```bash
plasflow2 run \
  --input        assembly.fasta \
  --output       ./results/ \
  --card-db      data/databases/card/card.dmnd \
  --aro-index    data/databases/card/aro_index.tsv \
  --sarg-db      data/databases/sarg/sarg.dmnd \
  --threads      8
```

---

## AMR risk score

Each plasmid contig receives a 0–10 risk score combining mobility, ARG burden, replicon breadth, and sample context:

| Factor | Score |
|---|---|
| Conjugative mobility | +3 |
| Mobilizable | +2 |
| ≥5 ARGs or ≥3 drug classes | +3 |
| 3–4 ARGs or 2 drug classes | +2 |
| 1–2 ARGs | +1 |
| Broad-host-range replicon (IncP / IncQ / IncW) | +2 |
| Known narrow-host-range replicon | +1 |
| Clinical or wastewater source (`--context`) | +2 |
| Environmental source | +1 |
| **Maximum (capped)** | **10** |

Risk ≥ 7 = high (red in report), 4–6 = medium (orange), 0–3 = low (green).

---

## Interactive HTML report

The HTML report (`report.html`) is fully self-contained — no server needed, open it in any browser. It includes:

- Summary stats panel (total sequences, plasmids, ARGs, taxonomy-classified contigs)
- Classification pie chart (Plotly)
- ARG counts by drug class (horizontal bar chart)
- Risk score distribution histogram (colour-coded by tier)
- **Contig length vs risk score scatter plot** (coloured by mobility class)
- **Top-15 taxonomy bar chart** for plasmid contigs (GTDB LCA assignments)
- **Risk-tier filter buttons** — show All / High (≥7) / Medium (4–6) / Low (0–3) contigs instantly
- Per-plasmid detail table with contig length, taxonomy (LCA), and full risk evidence (sortable + searchable via DataTables.js)

---

## Development

```bash
# Install dev dependencies
poetry install --with dev
pre-commit install

# Run all tests (175 tests — unit + integration)
pytest tests/ -v

# Lint + type-check
ruff check src/ tests/
black --check src/ tests/
mypy src/plasflow2/

# Fragment a large FASTA into contig-sized chunks for testing
python scripts/fragment_fasta.py \
  --input  genome.fasta \
  --output genome_contigs.fasta \
  --chunk  15000

# Download a real metagenome assembly for testing
python scripts/download_metagenome.py \
  --taxon wastewater \
  --min-size 150 \
  --outdir data/test/

# Download 1,000 diverse RefSeq chromosomes to retrain the classifier
# (balanced across 14 bacterial phyla via NCBI taxonomy ID search)
python scripts/download_refseq_chromosomes.py --outdir data/chromosomes/
# Smaller run for quick testing
python scripts/download_refseq_chromosomes.py --count 200 --outdir data/chromosomes/
# Dry run first to see what would be fetched
python scripts/download_refseq_chromosomes.py --dry-run --count 100
# Single phylum only
python scripts/download_refseq_chromosomes.py --phylum Pseudomonadota --count 100 --outdir data/chromosomes/
```

### Test structure

```
tests/
  unit/            153 tests — each module tested in isolation with synthetic data
    test_features.py       k-mer feature extraction (RC-aware)
    test_train.py          MLP + RF training pipeline
    test_build_dataset.py  dataset builder helpers
    test_fasta.py          FASTA I/O utilities
    test_args.py           ARG annotation (DIAMOND/CARD)
    test_mobility.py       MOB-suite integration
    test_risk_scorer.py    AMR risk scoring formula
    test_taxonomy.py       DIAMOND+LCA taxonomy (parse_lineage, lca_for_contig, assign_taxonomy)
    test_pipeline.py       end-to-end pipeline orchestration
    test_report.py         HTML report generator
    test_cli.py            CLI subcommands (CliRunner)
  integration/      22 tests — real data flow with mocked external binaries
    test_pipeline.py       classify→annotate→taxonomy→risk→report chain, all CLI subcommands
```

---

## Roadmap

- [x] Taxonomy annotation per contig (DIAMOND + GTDB LCA, Kaiju-style)
- [x] Drug-class co-occurrence heatmap across plasmid contigs (in report)
- [x] Docker image for zero-setup deployment
- [ ] Expanded chromosomal training data (1,000 species across 14 phyla — `scripts/download_refseq_chromosomes.py --count 1000`)
- [ ] Snakemake / Nextflow workflow wrapper for batch processing
- [ ] PyPI release

---

## Citation

If you use PlasFlow v2, please also cite the original PlasFlow:

> Krawczyk PS, Lipinski L, Dziembowski A. **PlasFlow: predicting plasmid sequences in metagenomic data using genome signatures.** *Nucleic Acids Research.* 2018;46(6):e35. doi:10.1093/nar/gkx1321

---

## License

GPL-3.0. See [LICENSE](LICENSE).
