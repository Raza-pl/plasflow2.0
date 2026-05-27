"""Unit tests for ARG annotation (annotate/args.py).

All tests use synthetic data — no DIAMOND, pyrodigal, or CARD files required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from plasflow2.annotate.args import load_card_metadata, parse_diamond_hits

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Minimal synthetic aro_index.tsv content matching real CARD column names
_ARO_INDEX_HEADER = (
    "ARO Accession\tCVTERM ID\tModel Sequence ID\tModel ID\tModel Name\t"
    "ARO Name\tProtein Accession\tDNA Accession\tAMR Gene Family\t"
    "Drug Class\tResistance Mechanism\tCARD Short Name\n"
)

_ARO_INDEX_ROWS = [
    "ARO:3002356\t38923\t123\t456\tNDM-6\tNDM-6\tAEX08599.1\tHM208592.1\t"
    "NDM beta-lactamase\tcarbapenem antibiotic; cephalosporin antibiotic\t"
    "antibiotic inactivation\tNDM-6\n",
    "ARO:3001109\t38456\t789\t101\tSHV-52\tSHV-52\tAEJ08681.1\tEF694408.1\t"
    "SHV beta-lactamase\tpenicillin antibiotic; cephalosporin antibiotic\t"
    "antibiotic inactivation\tSHV-52\n",
    "ARO:3002999\t43000\t200\t300\tCblA-1\tCblA-1\tACT97415.1\tGQ996933.1\t"
    "CblA carbapenemase\tcarbapenem antibiotic\tantibiotic inactivation\tCblA-1\n",
]

# Synthetic DIAMOND tabular output (format 6: qseqid sseqid pident qcovhsp evalue stitle)
_DIAMOND_TSV_LINES = [
    # NDM-6 hit — perfect match
    "contigA_1\tgb|AEX08599.1|ARO:3002356|NDM-6\t99.5\t95.2\t1e-120\t"
    "gb|AEX08599.1|ARO:3002356|NDM-6 [Escherichia coli]\n",
    # SHV-52 hit — barely above threshold
    "contigB_3\tgb|AEJ08681.1|ARO:3001109|SHV-52\t91.0\t82.0\t5e-80\t"
    "gb|AEJ08681.1|ARO:3001109|SHV-52 [Klebsiella pneumoniae]\n",
    # Malformed line — should be skipped
    "bad_line\n",
    # Line with only 5 fields — should be skipped
    "contigC_1\tgb|X\t90.0\t80.0\t1e-10\n",
]


def _write_aro_index(path: Path) -> None:
    path.write_text(_ARO_INDEX_HEADER + "".join(_ARO_INDEX_ROWS))


def _write_diamond_tsv(path: Path, lines: list[str] | None = None) -> None:
    path.write_text("".join(lines or _DIAMOND_TSV_LINES))


# ---------------------------------------------------------------------------
# load_card_metadata
# ---------------------------------------------------------------------------


def test_load_card_metadata_count(tmp_path: Path) -> None:
    p = tmp_path / "aro_index.tsv"
    _write_aro_index(p)
    meta = load_card_metadata(p)
    assert len(meta) == 3


def test_load_card_metadata_keys(tmp_path: Path) -> None:
    p = tmp_path / "aro_index.tsv"
    _write_aro_index(p)
    meta = load_card_metadata(p)
    assert "ARO:3002356" in meta
    assert "ARO:3001109" in meta
    assert "ARO:3002999" in meta


def test_load_card_metadata_fields(tmp_path: Path) -> None:
    p = tmp_path / "aro_index.tsv"
    _write_aro_index(p)
    meta = load_card_metadata(p)
    ndm = meta["ARO:3002356"]
    assert ndm["gene"] == "NDM-6"
    assert ndm["family"] == "NDM beta-lactamase"
    assert "carbapenem antibiotic" in ndm["drug_class"]
    assert ndm["mechanism"] == "antibiotic inactivation"


def test_load_card_metadata_semicolon_drug_class(tmp_path: Path) -> None:
    """Multi-drug class entries should be joined with '; '."""
    p = tmp_path / "aro_index.tsv"
    _write_aro_index(p)
    meta = load_card_metadata(p)
    shv = meta["ARO:3001109"]
    assert "penicillin antibiotic" in shv["drug_class"]
    assert "cephalosporin antibiotic" in shv["drug_class"]


def test_load_card_metadata_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "aro_index.tsv"
    p.write_text(_ARO_INDEX_HEADER)  # header only, no data rows
    meta = load_card_metadata(p)
    assert meta == {}


# ---------------------------------------------------------------------------
# parse_diamond_hits
# ---------------------------------------------------------------------------


def test_parse_hits_count(tmp_path: Path) -> None:
    tsv = tmp_path / "diamond.tsv"
    _write_diamond_tsv(tsv)
    hits = parse_diamond_hits(tsv)
    assert len(hits) == 2  # 2 valid lines; 2 malformed skipped


def test_parse_hits_contig_id_strips_orf_suffix(tmp_path: Path) -> None:
    tsv = tmp_path / "diamond.tsv"
    _write_diamond_tsv(tsv)
    hits = parse_diamond_hits(tsv)
    assert hits[0].contig_id == "contigA"
    assert hits[1].contig_id == "contigB"


def test_parse_hits_aro_accession(tmp_path: Path) -> None:
    tsv = tmp_path / "diamond.tsv"
    _write_diamond_tsv(tsv)
    hits = parse_diamond_hits(tsv)
    assert hits[0].aro_accession == "ARO:3002356"
    assert hits[1].aro_accession == "ARO:3001109"


def test_parse_hits_gene_name_from_sseqid(tmp_path: Path) -> None:
    """Gene name should be parsed from sseqid even without metadata dict."""
    tsv = tmp_path / "diamond.tsv"
    _write_diamond_tsv(tsv)
    hits = parse_diamond_hits(tsv)
    assert hits[0].gene_name == "NDM-6"
    assert hits[1].gene_name == "SHV-52"


def test_parse_hits_with_metadata_enrichment(tmp_path: Path) -> None:
    tsv = tmp_path / "diamond.tsv"
    aro = tmp_path / "aro_index.tsv"
    _write_diamond_tsv(tsv)
    _write_aro_index(aro)
    metadata = load_card_metadata(aro)
    hits = parse_diamond_hits(tsv, metadata)
    ndm = hits[0]
    assert ndm.amr_family == "NDM beta-lactamase"
    assert "carbapenem antibiotic" in ndm.drug_class
    assert ndm.resistance_mechanism == "antibiotic inactivation"


def test_parse_hits_numeric_fields(tmp_path: Path) -> None:
    tsv = tmp_path / "diamond.tsv"
    _write_diamond_tsv(tsv)
    hits = parse_diamond_hits(tsv)
    assert hits[0].identity == pytest.approx(99.5)
    assert hits[0].coverage == pytest.approx(95.2)
    assert hits[0].evalue == pytest.approx(1e-120)


def test_parse_hits_empty_tsv(tmp_path: Path) -> None:
    tsv = tmp_path / "diamond.tsv"
    tsv.write_text("")
    hits = parse_diamond_hits(tsv)
    assert hits == []


def test_parse_hits_unknown_aro_fallback(tmp_path: Path) -> None:
    """sseqid with non-CARD format should produce 'unknown' ARO but not crash."""
    tsv = tmp_path / "diamond.tsv"
    tsv.write_text("contigX_2\tunknown_db|ACCXYZ\t95.0\t88.0\t1e-50\tsome gene\n")
    hits = parse_diamond_hits(tsv)
    assert len(hits) == 1
    assert hits[0].aro_accession == "unknown"
    assert hits[0].contig_id == "contigX"
