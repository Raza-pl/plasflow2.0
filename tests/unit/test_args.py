"""Unit tests for ARG annotation (annotate/args.py).

Covers both CARD-only and dual CARD+SARG annotation paths.
All tests use synthetic data — no DIAMOND, pyrodigal, or database files required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from plasflow2.annotate.args import (
    ARGHit,
    load_card_metadata,
    merge_arg_hits,
    parse_diamond_hits,
    parse_sarg_hits,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers — CARD
# ---------------------------------------------------------------------------

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
# Fixtures / helpers — SARG
# ---------------------------------------------------------------------------

# SARG DIAMOND output: sseqid uses actual SARG format SARG|drug_type|gene_family*|WP_accession
_SARG_TSV_LINES = [
    # mcr — polymyxin resistance, not in CARD synthetic set
    "contigC_2\tSARG|polymyxin|mcr*|WP_000001234.1\t85.0\t82.0\t2e-60\t"
    "SARG|polymyxin|mcr*|WP_000001234.1 mobile colistin resistance protein\n",
    # aph(6) — aminoglycoside, SARG-only contig
    "contigD_1\tSARG|aminoglycoside|aph(6)*|WP_000005678.1\t88.5\t90.0\t3e-75\t"
    "SARG|aminoglycoside|aph(6)*|WP_000005678.1 aminoglycoside phosphotransferase\n",
    # Same ORF as CARD contigA_1 — should be deduplicated in favour of CARD
    "contigA_1\tSARG|beta-lactam|bla*|WP_459377734.1\t81.0\t80.0\t5e-40\t"
    "SARG|beta-lactam|bla*|WP_459377734.1 class A beta-lactamase\n",
    # Malformed — skip
    "bad\n",
]


def _write_sarg_tsv(path: Path, lines: list[str] | None = None) -> None:
    path.write_text("".join(lines or _SARG_TSV_LINES))


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
# parse_diamond_hits (CARD)
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


def test_parse_hits_source_is_card(tmp_path: Path) -> None:
    """All hits from parse_diamond_hits should have source='CARD'."""
    tsv = tmp_path / "diamond.tsv"
    _write_diamond_tsv(tsv)
    hits = parse_diamond_hits(tsv)
    assert all(h.source == "CARD" for h in hits)


# ---------------------------------------------------------------------------
# parse_sarg_hits
# ---------------------------------------------------------------------------


def test_parse_sarg_hits_count(tmp_path: Path) -> None:
    tsv = tmp_path / "sarg.tsv"
    _write_sarg_tsv(tsv)
    hits = parse_sarg_hits(tsv)
    # 3 valid lines (mcr-1, aph, and contigA_1 duplicate), 1 malformed
    assert len(hits) == 3


def test_parse_sarg_hits_source(tmp_path: Path) -> None:
    """All hits from parse_sarg_hits should have source='SARG'."""
    tsv = tmp_path / "sarg.tsv"
    _write_sarg_tsv(tsv)
    hits = parse_sarg_hits(tsv)
    assert all(h.source == "SARG" for h in hits)


def test_parse_sarg_hits_drug_class_from_header(tmp_path: Path) -> None:
    """Drug class should be extracted from the pipe-delimited SARG header."""
    tsv = tmp_path / "sarg.tsv"
    _write_sarg_tsv(tsv)
    hits = parse_sarg_hits(tsv)
    # mcr-1 → polymyxin; aph → aminoglycoside
    drug_classes = {h.drug_class for h in hits}
    assert "polymyxin" in drug_classes
    assert "aminoglycoside" in drug_classes


def test_parse_sarg_hits_gene_name(tmp_path: Path) -> None:
    """Gene name should be the gene_family field (trailing * stripped)."""
    tsv = tmp_path / "sarg.tsv"
    _write_sarg_tsv(tsv)
    hits = parse_sarg_hits(tsv)
    gene_names = {h.gene_name for h in hits}
    assert "mcr" in gene_names
    assert "aph(6)" in gene_names


def test_parse_sarg_hits_amr_family_is_gene_family(tmp_path: Path) -> None:
    """amr_family should be the SARG gene_family field (trailing * stripped)."""
    tsv = tmp_path / "sarg.tsv"
    _write_sarg_tsv(tsv)
    hits = parse_sarg_hits(tsv)
    subtypes = {h.amr_family for h in hits}
    assert "mcr" in subtypes
    assert "aph(6)" in subtypes


def test_parse_sarg_hits_contig_strips_orf_suffix(tmp_path: Path) -> None:
    tsv = tmp_path / "sarg.tsv"
    _write_sarg_tsv(tsv)
    hits = parse_sarg_hits(tsv)
    contig_ids = {h.contig_id for h in hits}
    assert "contigC" in contig_ids
    assert "contigD" in contig_ids


def test_parse_sarg_hits_empty_file(tmp_path: Path) -> None:
    tsv = tmp_path / "sarg.tsv"
    tsv.write_text("")
    hits = parse_sarg_hits(tsv)
    assert hits == []


def test_parse_sarg_hits_fallback_no_pipes(tmp_path: Path) -> None:
    """sseqid without pipe delimiters should not crash — falls back gracefully."""
    tsv = tmp_path / "sarg.tsv"
    tsv.write_text("contigZ_1\tnopipes\t82.0\t85.0\t1e-30\tnopipes description\n")
    hits = parse_sarg_hits(tsv)
    assert len(hits) == 1
    assert hits[0].gene_name == "nopipes"
    assert hits[0].drug_class == "unknown"
    assert hits[0].source == "SARG"


# ---------------------------------------------------------------------------
# merge_arg_hits
# ---------------------------------------------------------------------------


def test_merge_card_preferred_per_orf(tmp_path: Path) -> None:
    """If same ORF appears in both CARD and SARG, the CARD hit is kept."""
    card = [
        ARGHit(
            contig_id="contigA",
            gene_name="NDM-6",
            aro_accession="ARO:3002356",
            amr_family="NDM beta-lactamase",
            drug_class="carbapenem antibiotic",
            resistance_mechanism="antibiotic inactivation",
            identity=99.5,
            coverage=95.0,
            evalue=1e-120,
            source="CARD",
            _orf_id="contigA_1",
        )
    ]
    sarg = [
        ARGHit(
            contig_id="contigA",
            gene_name="NDM-1",
            aro_accession="SARG_NDM",
            amr_family="NDM",
            drug_class="beta-lactam",
            resistance_mechanism="unknown",
            identity=81.0,
            coverage=80.0,
            evalue=5e-40,
            source="SARG",
            _orf_id="contigA_1",  # same ORF → should be dropped
        )
    ]
    merged = merge_arg_hits(card, sarg)
    assert len(merged) == 1
    assert merged[0].source == "CARD"
    assert merged[0].gene_name == "NDM-6"


def test_merge_sarg_only_supplemented(tmp_path: Path) -> None:
    """SARG hits for ORFs not found by CARD are added to the merged list."""
    card = [
        ARGHit(
            contig_id="contigA",
            gene_name="NDM-6",
            aro_accession="ARO:3002356",
            amr_family="NDM beta-lactamase",
            drug_class="carbapenem antibiotic",
            resistance_mechanism="antibiotic inactivation",
            identity=99.5,
            coverage=95.0,
            evalue=1e-120,
            source="CARD",
            _orf_id="contigA_1",
        )
    ]
    sarg = [
        ARGHit(
            contig_id="contigB",
            gene_name="mcr-1",
            aro_accession="Ec_mcr1",
            amr_family="MCR",
            drug_class="polymyxin",
            resistance_mechanism="unknown",
            identity=85.0,
            coverage=82.0,
            evalue=2e-60,
            source="SARG",
            _orf_id="contigB_1",  # different ORF — should be kept
        )
    ]
    merged = merge_arg_hits(card, sarg)
    assert len(merged) == 2
    sources = {h.source for h in merged}
    assert "CARD" in sources
    assert "SARG" in sources


def test_merge_empty_sarg(tmp_path: Path) -> None:
    """merge_arg_hits with empty SARG list returns all CARD hits unchanged."""
    card = [
        ARGHit(
            contig_id="contigA",
            gene_name="NDM-6",
            aro_accession="ARO:3002356",
            amr_family="NDM beta-lactamase",
            drug_class="carbapenem antibiotic",
            resistance_mechanism="antibiotic inactivation",
            identity=99.5,
            coverage=95.0,
            evalue=1e-120,
            source="CARD",
            _orf_id="contigA_1",
        )
    ]
    merged = merge_arg_hits(card, [])
    assert merged == card


def test_merge_empty_card(tmp_path: Path) -> None:
    """merge_arg_hits with empty CARD list returns all SARG hits."""
    sarg = [
        ARGHit(
            contig_id="contigC",
            gene_name="mcr-1",
            aro_accession="Ec_mcr1",
            amr_family="MCR",
            drug_class="polymyxin",
            resistance_mechanism="unknown",
            identity=85.0,
            coverage=82.0,
            evalue=2e-60,
            source="SARG",
            _orf_id="contigC_2",
        )
    ]
    merged = merge_arg_hits([], sarg)
    assert len(merged) == 1
    assert merged[0].source == "SARG"


def test_merge_both_empty() -> None:
    assert merge_arg_hits([], []) == []


def test_merge_preserves_order(tmp_path: Path) -> None:
    """CARD hits come first; SARG-only appended in their original order."""
    card = [
        ARGHit(
            contig_id=f"c{i}",
            gene_name=f"gene{i}",
            aro_accession=f"ARO:{i}",
            amr_family="fam",
            drug_class="dc",
            resistance_mechanism="inactivation",
            identity=95.0,
            coverage=90.0,
            evalue=1e-50,
            source="CARD",
            _orf_id=f"c{i}_1",
        )
        for i in range(3)
    ]
    sarg = [
        ARGHit(
            contig_id=f"s{i}",
            gene_name=f"sarg{i}",
            aro_accession=f"S{i}",
            amr_family="fam",
            drug_class="dc",
            resistance_mechanism="unknown",
            identity=82.0,
            coverage=80.0,
            evalue=1e-30,
            source="SARG",
            _orf_id=f"s{i}_1",
        )
        for i in range(2)
    ]
    merged = merge_arg_hits(card, sarg)
    assert [h.source for h in merged] == ["CARD", "CARD", "CARD", "SARG", "SARG"]


# ---------------------------------------------------------------------------
# ARGHit.source default
# ---------------------------------------------------------------------------


def test_arg_hit_default_source() -> None:
    hit = ARGHit(
        contig_id="c",
        gene_name="g",
        aro_accession="ARO:1",
        amr_family="fam",
        drug_class="dc",
        resistance_mechanism="inactivation",
        identity=95.0,
        coverage=90.0,
        evalue=1e-50,
    )
    assert hit.source == "CARD"
