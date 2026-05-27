"""Unit tests for MOB-suite mobility annotation (annotate/mobility.py).

All tests use synthetic data — no mob_typer binary required.
"""

from __future__ import annotations

from pathlib import Path

from plasflow2.annotate.mobility import (
    MOBILITY_CLASSES,
    index_by_contig,
    parse_mob_results,
)

# ---------------------------------------------------------------------------
# Synthetic mob_typer 3.x TSV content
# ---------------------------------------------------------------------------

_HEADER = (
    "sample_id\tnum_contigs\tsize\tgc\tmd5\t"
    "rep_type(s)\trep_type_accession(s)\t"
    "relaxase_type(s)\trelaxase_type_accession(s)\t"
    "mpf_type\tmpf_type_accession(s)\t"
    "orit_type(s)\torit_accession(s)\t"
    "predicted_mobility\tmash_nearest_neighbor\tmash_neighbor_distance\n"
)

# Three representative rows: conjugative, mobilizable, non-mobilizable
_ROW_CONJ = (
    "plasmid_A\t1\t52000\t0.54\tabc123\t"
    "IncP-1alpha\tAP002527\t"
    "MOBP\tCP000828\t"
    "MPF_T\tCP000828\t"
    "-\t-\t"
    "conjugative\tNZ_CP000828.1\t0.001\n"
)
_ROW_MOB = (
    "plasmid_B\t1\t8000\t0.48\tdef456\t"
    "Col(pHAD28)\tCP012142\t"
    "MOBQ\tAB286500\t"
    "-\t-\t"
    "-\t-\t"
    "mobilizable\tNZ_CP012142.1\t0.002\n"
)
_ROW_NONMOB = (
    "plasmid_C\t1\t3500\t0.45\tghi789\t"
    "-\t-\t"
    "-\t-\t"
    "-\t-\t"
    "-\t-\t"
    "non-mobilizable\tNZ_X12345.1\t0.010\n"
)

# Row with an unrecognised mobility class — should fall back to non-mobilizable
_ROW_UNKNOWN_MOB = (
    "plasmid_D\t1\t1000\t0.50\tjkl000\t"
    "-\t-\t"
    "-\t-\t"
    "-\t-\t"
    "-\t-\t"
    "UNKNOWN_CLASS\tNZ_X99999.1\t0.500\n"
)


def _write_tsv(path: Path, rows: list[str]) -> None:
    path.write_text(_HEADER + "".join(rows))


# ---------------------------------------------------------------------------
# parse_mob_results — basic parsing
# ---------------------------------------------------------------------------


def test_parse_count(tmp_path: Path) -> None:
    tsv = tmp_path / "mobtyper_results.txt"
    _write_tsv(tsv, [_ROW_CONJ, _ROW_MOB, _ROW_NONMOB])
    results = parse_mob_results(tsv)
    assert len(results) == 3


def test_parse_contig_ids(tmp_path: Path) -> None:
    tsv = tmp_path / "mobtyper_results.txt"
    _write_tsv(tsv, [_ROW_CONJ, _ROW_MOB, _ROW_NONMOB])
    results = parse_mob_results(tsv)
    ids = [r.contig_id for r in results]
    assert ids == ["plasmid_A", "plasmid_B", "plasmid_C"]


def test_parse_mobility_classes(tmp_path: Path) -> None:
    tsv = tmp_path / "mobtyper_results.txt"
    _write_tsv(tsv, [_ROW_CONJ, _ROW_MOB, _ROW_NONMOB])
    results = parse_mob_results(tsv)
    assert results[0].mobility_class == "conjugative"
    assert results[1].mobility_class == "mobilizable"
    assert results[2].mobility_class == "non-mobilizable"


def test_parse_replicon_type(tmp_path: Path) -> None:
    tsv = tmp_path / "mobtyper_results.txt"
    _write_tsv(tsv, [_ROW_CONJ])
    results = parse_mob_results(tsv)
    assert results[0].replicon_type == "IncP-1alpha"


def test_parse_relaxase_type(tmp_path: Path) -> None:
    tsv = tmp_path / "mobtyper_results.txt"
    _write_tsv(tsv, [_ROW_CONJ])
    results = parse_mob_results(tsv)
    assert results[0].relaxase_type == "MOBP"


def test_parse_mpf_type(tmp_path: Path) -> None:
    tsv = tmp_path / "mobtyper_results.txt"
    _write_tsv(tsv, [_ROW_CONJ])
    results = parse_mob_results(tsv)
    assert results[0].mpf_type == "MPF_T"


# ---------------------------------------------------------------------------
# parse_mob_results — normalisation
# ---------------------------------------------------------------------------


def test_dash_replicon_becomes_unknown(tmp_path: Path) -> None:
    """A '-' replicon_type should be normalised to 'unknown'."""
    tsv = tmp_path / "mobtyper_results.txt"
    _write_tsv(tsv, [_ROW_NONMOB])
    results = parse_mob_results(tsv)
    assert results[0].replicon_type == "unknown"


def test_dash_relaxase_becomes_none(tmp_path: Path) -> None:
    tsv = tmp_path / "mobtyper_results.txt"
    _write_tsv(tsv, [_ROW_NONMOB])
    results = parse_mob_results(tsv)
    assert results[0].relaxase_type == "none"


def test_dash_mpf_becomes_none(tmp_path: Path) -> None:
    tsv = tmp_path / "mobtyper_results.txt"
    _write_tsv(tsv, [_ROW_MOB])
    results = parse_mob_results(tsv)
    assert results[1 - 1].mpf_type == "none"  # plasmid_B has no MPF


def test_unknown_mobility_class_falls_back(tmp_path: Path) -> None:
    """Unrecognised mobility class should fall back to 'non-mobilizable'."""
    tsv = tmp_path / "mobtyper_results.txt"
    _write_tsv(tsv, [_ROW_UNKNOWN_MOB])
    results = parse_mob_results(tsv)
    assert len(results) == 1
    assert results[0].mobility_class == "non-mobilizable"


def test_all_mobility_classes_valid(tmp_path: Path) -> None:
    """All parsed mobility_class values should be members of MOBILITY_CLASSES."""
    tsv = tmp_path / "mobtyper_results.txt"
    _write_tsv(tsv, [_ROW_CONJ, _ROW_MOB, _ROW_NONMOB])
    results = parse_mob_results(tsv)
    for r in results:
        assert r.mobility_class in MOBILITY_CLASSES


# ---------------------------------------------------------------------------
# parse_mob_results — edge cases
# ---------------------------------------------------------------------------


def test_header_only_returns_empty(tmp_path: Path) -> None:
    """A file with only the header row should produce an empty list."""
    tsv = tmp_path / "mobtyper_results.txt"
    tsv.write_text(_HEADER)
    results = parse_mob_results(tsv)
    assert results == []


def test_completely_empty_file_returns_empty(tmp_path: Path) -> None:
    tsv = tmp_path / "mobtyper_results.txt"
    tsv.write_text("")
    results = parse_mob_results(tsv)
    assert results == []


def test_raw_dict_preserved(tmp_path: Path) -> None:
    """raw field should contain all columns from the header."""
    tsv = tmp_path / "mobtyper_results.txt"
    _write_tsv(tsv, [_ROW_CONJ])
    results = parse_mob_results(tsv)
    assert "mash_nearest_neighbor" in results[0].raw
    assert results[0].raw["mash_nearest_neighbor"] == "NZ_CP000828.1"


# ---------------------------------------------------------------------------
# index_by_contig
# ---------------------------------------------------------------------------


def test_index_by_contig_keys(tmp_path: Path) -> None:
    tsv = tmp_path / "mobtyper_results.txt"
    _write_tsv(tsv, [_ROW_CONJ, _ROW_MOB, _ROW_NONMOB])
    results = parse_mob_results(tsv)
    idx = index_by_contig(results)
    assert set(idx.keys()) == {"plasmid_A", "plasmid_B", "plasmid_C"}


def test_index_by_contig_values(tmp_path: Path) -> None:
    tsv = tmp_path / "mobtyper_results.txt"
    _write_tsv(tsv, [_ROW_CONJ])
    results = parse_mob_results(tsv)
    idx = index_by_contig(results)
    assert idx["plasmid_A"].mobility_class == "conjugative"


def test_index_by_contig_empty(tmp_path: Path) -> None:
    assert index_by_contig([]) == {}
