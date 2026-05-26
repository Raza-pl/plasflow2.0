# PlasFlow v2

[![CI](https://github.com/YOUR_USERNAME/plasflow2/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/plasflow2/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**PlasFlow v2** classifies metagenomic contigs as plasmid, chromosome, phage, or archaea, annotates antibiotic resistance genes (ARGs) via CARD, determines mobility class with MOB-suite, and produces an interactive HTML report — all in one command.

This is a full rewrite of [PlasFlow v1](https://github.com/smaegol/PlasFlow) (Krawczyk et al., *Nucleic Acids Research* 2018) on a modern stack.

---

## What's new in v2

| Feature | v1 | v2 |
|---|---|---|
| Python version | 3.5 / TensorFlow 0.10 | 3.10+ / PyTorch 2.x |
| Classes | plasmid vs chromosome | plasmid · chromosome · **phage** · **archaea** |
| ARG annotation | ✗ | DIAMOND + CARD |
| Mobility typing | ✗ | MOB-suite |
| AMR risk score | ✗ | 0–10 score with evidence |
| Output | TSV only | TSV + FASTA bins + **interactive HTML report** |
| Install | conda-only | **pip-installable**, Docker image |
| Apple Silicon | ✗ | MPS (M1/M2/M3) accelerated |

---

## Installation

### pip (recommended)

```bash
pip install plasflow2
```

### From source (development)

```bash
git clone https://github.com/YOUR_USERNAME/plasflow2
cd plasflow2
pip install poetry
poetry install
```

### Apple Silicon (M1/M2/M3)

```bash
CONDA_SUBDIR=osx-arm64 conda create -n plasflow2 python=3.10
conda activate plasflow2
pip install torch torchvision torchaudio   # MPS-enabled by default on Apple Silicon
pip install plasflow2
```

### Docker

```bash
docker pull ghcr.io/YOUR_USERNAME/plasflow2:latest
docker run --rm \
  -v /path/to/databases:/data/databases:ro \
  -v $(pwd)/results:/work/results \
  plasflow2 run --input /work/assembly.fasta --output /work/results
```

---

## Download databases

```bash
# Downloads PLSDB, CARD, INPHARED (~10 GB total — run on Day 1)
python scripts/download_databases.py
```

> **Note:** PLSDB (~8 GB) and INPHARED (~30k genomes) are large. Start downloads early; they run in the background on the CPU machine.

---

## Quickstart

```bash
# Full pipeline: classify + annotate + risk score + HTML report
plasflow2 run \
  --input assembly.fasta \
  --output ./results/ \
  --threshold 0.7 \
  --context clinical \
  --threads 8
```

**Output files in `./results/`:**

| File | Description |
|---|---|
| `plasmids.fasta` | Classified plasmid sequences |
| `chromosomes.fasta` | Classified chromosomal sequences |
| `phages.fasta` | Classified phage sequences |
| `archaea.fasta` | Classified archaeal sequences |
| `unclassified.fasta` | Low-confidence sequences (< threshold) |
| `predictions.tsv` | Per-sequence classification + confidence |
| `annotations.json` | ARG hits, mobility type, replicon, PLSDB match |
| `report.html` | Self-contained interactive HTML report |

---

## CLI reference

```
plasflow2 run       Full pipeline (classify + annotate + risk + report)
plasflow2 classify  Classify sequences only → predictions.tsv
plasflow2 annotate  Annotate plasmids with ARGs and mobility
plasflow2 risk      Compute AMR risk scores from annotation data

Options (plasflow2 run):
  --input / -i        Input assembly FASTA                [required]
  --output / -o       Output directory                    [required]
  --model             Path to custom .pt model weights
  --threshold         Confidence threshold (default: 0.7)
  --context           Sample source: clinical | wastewater | environmental | unspecified
  --threads           CPU threads for DIAMOND/BLAST       (default: 8)
  --min-length        Minimum contig length in bp         (default: 1000)
```

---

## AMR risk score

Each plasmid receives a 0–10 risk score:

| Factor | Score |
|---|---|
| Conjugative mobility | +3 |
| Mobilizable | +2 |
| ≥5 ARGs or ≥3 drug classes | +3 |
| 3–4 ARGs or 2 drug classes | +2 |
| 1–2 ARGs | +1 |
| Broad-host-range replicon (IncP/Q/W) | +2 |
| Known narrow-host-range replicon | +1 |
| Clinical / wastewater source | +2 |
| Environmental source | +1 |
| **Maximum** | **10** |

Use `--context clinical` (or `wastewater` / `environmental`) to include sample-source evidence.

---

## Benchmark vs v1

*(To be updated on Day 28 with final results)*

| Tool | Plasmid F1 | Chromosome F1 | Phage F1 | Archaea F1 | Runtime / Mb |
|---|---|---|---|---|---|
| PlasFlow v1 | ~0.97 | ~0.96 | — | — | ~30 s |
| PlasClass | ~0.95 | ~0.94 | — | — | — |
| **PlasFlow v2** | TBD | TBD | TBD | TBD | TBD |

---

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/
black --check src/ tests/

# Install pre-commit hooks
pre-commit install
```

---

## License

GPL-3.0. See [LICENSE](LICENSE).

Based on PlasFlow by Krawczyk et al., *Nucleic Acids Research* 2018.
