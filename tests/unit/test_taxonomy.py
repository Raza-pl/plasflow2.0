"""Unit tests for plasflow2.annotate.taxonomy.

Tests cover:
  - parse_lineage()
  - _extract_lineage_from_stitle()
  - lca_for_contig()
  - parse_diamond_taxonomy_output()
  - build_gtdb_taxon_map() / load_taxon_map()
  - assign_taxonomy() (with mocked run_diamond_taxonomy)
  - summarise_taxonomy()
  - TaxResult.display / TaxResult.lineage_dict
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from plasflow2.annotate.taxonomy import (
    TaxHit,
    TaxResult,
    _extract_lineage_from_stitle,
    assign_taxonomy,
    build_gtdb_taxon_map,
    lca_for_contig,
    load_taxon_map,
    parse_diamond_taxonomy_output,
    parse_lineage,
    summarise_taxonomy,
)

# ---------------------------------------------------------------------------
# Fixtures — shared GTDB lineage strings
# ---------------------------------------------------------------------------

LIN_ECOLI = "d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli"
LIN_KPNEU = "d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Klebsiella;s__Klebsiella pneumoniae"
LIN_BSUB = "d__Bacteria;p__Bacillota;c__Bacilli;o__Bacillales;f__Bacillaceae;g__Bacillus;s__Bacillus subtilis"
LIN_ARCH = "d__Archaea;p__Halobacteriota;c__Halobacteria;o__Halobacteriales;f__Halobacteriaceae;g__Halobacterium;s__Halobacterium salinarum"


# ---------------------------------------------------------------------------
# parse_lineage
# ---------------------------------------------------------------------------


class TestParseLineage:
    def test_full_gtdb_lineage(self):
        result = parse_lineage(LIN_ECOLI)
        assert len(result) == 7
        prefixes = [p for p, _ in result]
        assert prefixes == ["d__", "p__", "c__", "o__", "f__", "g__", "s__"]

    def test_partial_lineage(self):
        lin = "d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria"
        result = parse_lineage(lin)
        assert len(result) == 3
        assert result[2] == ("c__", "c__Gammaproteobacteria")

    def test_empty_levels_are_skipped(self):
        lin = "d__Bacteria;p__;c__Gammaproteobacteria"
        result = parse_lineage(lin)
        prefixes = [p for p, _ in result]
        assert "p__" not in prefixes  # empty name → skipped

    def test_unknown_placeholder_skipped(self):
        lin = "d__Bacteria;p__unknown;c__Gammaproteobacteria"
        result = parse_lineage(lin)
        prefixes = [p for p, _ in result]
        assert "p__" not in prefixes

    def test_empty_string(self):
        assert parse_lineage("") == []

    def test_no_recognisable_prefixes(self):
        assert parse_lineage("SomeRandomString;AnotherThing") == []


# ---------------------------------------------------------------------------
# _extract_lineage_from_stitle
# ---------------------------------------------------------------------------


class TestExtractLineageFromStitle:
    def test_lineage_in_stitle(self):
        stitle = "WP_012345678.1 protein_name [Escherichia coli] d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria"
        result = _extract_lineage_from_stitle(stitle)
        assert result.startswith("d__Bacteria")

    def test_no_lineage_returns_empty(self):
        assert _extract_lineage_from_stitle("WP_012345678.1 some protein") == ""

    def test_lineage_only(self):
        result = _extract_lineage_from_stitle(LIN_ECOLI)
        assert result == LIN_ECOLI


# ---------------------------------------------------------------------------
# lca_for_contig
# ---------------------------------------------------------------------------


def _make_hit(contig_id: str, lineage: str, bitscore: float = 500.0) -> TaxHit:
    return TaxHit(
        contig_id=contig_id,
        accession="ACC_001",
        lineage=lineage,
        identity=95.0,
        coverage=90.0,
        evalue=1e-80,
        bit_score=bitscore,
    )


class TestLcaForContig:
    def test_no_hits_returns_unclassified(self):
        result = lca_for_contig([])
        assert result.rank == "unclassified"
        assert result.taxon == ""

    def test_all_hits_agree_at_species(self):
        hits = [_make_hit("c1", LIN_ECOLI) for _ in range(5)]
        result = lca_for_contig(hits)
        assert result.rank == "species"
        assert result.taxon == "s__Escherichia coli"

    def test_mixed_species_same_genus_lca_at_genus(self):
        # 6 E. coli + 4 E. fergusonii → genus Escherichia should win (100% genus agreement)
        lin_eferg = "d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia fergusonii"
        hits = [_make_hit("c2", LIN_ECOLI) for _ in range(6)] + [
            _make_hit("c2", lin_eferg) for _ in range(4)
        ]
        result = lca_for_contig(hits)
        # 100% agree at genus (all Escherichia) → should go to genus or deeper
        assert result.rank in ("genus", "species")
        assert "Escherichia" in result.taxon

    def test_cross_genus_lca_at_family(self):
        # 5 Escherichia + 5 Klebsiella: exact 50/50 split at genus level.
        # Strict majority (>) means 5/10 = 0.5 does NOT pass, so LCA stops at
        # family (Enterobacteriaceae), where both share 100% agreement.
        hits = [_make_hit("c3", LIN_ECOLI) for _ in range(5)] + [
            _make_hit("c3", LIN_KPNEU) for _ in range(5)
        ]
        result = lca_for_contig(hits)
        assert result.rank == "family"
        assert result.taxon == "f__Enterobacteriaceae"

    def test_cross_phylum_lca_at_domain(self):
        # 5 Proteobacteria + 5 Bacillota: exact tie at phylum, both 5/10 = 0.5.
        # Strict majority stops at domain (d__Bacteria, 10/10 = 1.0).
        hits = [_make_hit("c4", LIN_ECOLI) for _ in range(5)] + [
            _make_hit("c4", LIN_BSUB) for _ in range(5)
        ]
        result = lca_for_contig(hits)
        assert result.rank == "domain"
        assert result.taxon == "d__Bacteria"

    def test_bacteria_vs_archaea_majority_wins_to_species(self):
        # 4 E.coli + 6 Halobacterium: Archaea wins with 60% at every rank
        # (all 6 Archaea hits share the same species), so LCA reaches species.
        hits = [_make_hit("c5", LIN_ECOLI) for _ in range(4)] + [
            _make_hit("c5", LIN_ARCH) for _ in range(6)
        ]
        result = lca_for_contig(hits)
        # 6/10 = 0.6 > 0.5 at every rank down to species → species wins
        assert result.rank == "species"
        assert "Halobacterium" in result.taxon

    def test_no_lineage_in_hits(self):
        hits = [_make_hit("c6", "") for _ in range(5)]
        result = lca_for_contig(hits)
        assert result.rank == "unclassified"

    def test_custom_min_agreement(self):
        # At 80% threshold, 6/10 Escherichia hits should not reach species
        lin_eferg = "d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia fergusonii"
        hits = [_make_hit("c7", LIN_ECOLI) for _ in range(6)] + [
            _make_hit("c7", lin_eferg) for _ in range(4)
        ]
        result = lca_for_contig(hits, min_agreement=0.8)
        # 6/10 = 60% agree at species — below 0.8, so falls back to genus level (100%)
        assert result.rank in ("genus",)


# ---------------------------------------------------------------------------
# parse_diamond_taxonomy_output
# ---------------------------------------------------------------------------


class TestParseDiamondTaxonomyOutput:
    def _write_tsv(self, tmp_path: Path, rows: list[list]) -> Path:
        tsv = tmp_path / "diamond_taxonomy.tsv"
        with open(tsv, "w") as fh:
            for row in rows:
                fh.write("\t".join(str(c) for c in row) + "\n")
        return tsv

    def test_basic_parsing(self, tmp_path):
        rows = [
            ["contig1", "ACC001", "95.0", "90.0", "1e-80", "500.0", LIN_ECOLI],
            ["contig1", "ACC002", "92.0", "88.0", "1e-75", "480.0", LIN_ECOLI],
            ["contig2", "ACC003", "90.0", "85.0", "1e-70", "460.0", LIN_BSUB],
        ]
        tsv = self._write_tsv(tmp_path, rows)
        result = parse_diamond_taxonomy_output(tsv)
        assert "contig1" in result
        assert "contig2" in result
        assert len(result["contig1"]) == 2
        assert result["contig1"][0].bit_score >= result["contig1"][1].bit_score

    def test_lineage_from_stitle(self, tmp_path):
        rows = [["c1", "ACC001", "95.0", "90.0", "1e-80", "500.0", LIN_ECOLI]]
        tsv = self._write_tsv(tmp_path, rows)
        result = parse_diamond_taxonomy_output(tsv)
        assert result["c1"][0].lineage == LIN_ECOLI

    def test_lineage_from_taxon_map(self, tmp_path):
        rows = [["c1", "GCF_000001405", "95.0", "90.0", "1e-80", "500.0", "no lineage here"]]
        tsv = self._write_tsv(tmp_path, rows)
        taxon_map = {"GCF_000001405": LIN_ECOLI}
        result = parse_diamond_taxonomy_output(tsv, taxon_map=taxon_map)
        assert result["c1"][0].lineage == LIN_ECOLI

    def test_top_n_limit(self, tmp_path):
        rows = [
            ["c1", f"ACC{i:03d}", "90.0", "85.0", f"1e-{60+i}", str(600 - i * 10), LIN_ECOLI]
            for i in range(20)
        ]
        tsv = self._write_tsv(tmp_path, rows)
        result = parse_diamond_taxonomy_output(tsv, top_n=5)
        assert len(result["c1"]) == 5

    def test_orf_suffix_stripped(self, tmp_path):
        # ORF IDs like "contig1_3" should map back to "contig1"
        rows = [["contig1_3", "ACC001", "95.0", "90.0", "1e-80", "500.0", LIN_ECOLI]]
        tsv = self._write_tsv(tmp_path, rows)
        result = parse_diamond_taxonomy_output(tsv)
        assert "contig1" in result

    def test_empty_file(self, tmp_path):
        tsv = tmp_path / "empty.tsv"
        tsv.write_text("")
        result = parse_diamond_taxonomy_output(tsv)
        assert result == {}

    def test_comment_lines_skipped(self, tmp_path):
        rows_text = "# comment line\nc1\tACC001\t95.0\t90.0\t1e-80\t500.0\t" + LIN_ECOLI + "\n"
        tsv = tmp_path / "diamond_taxonomy.tsv"
        tsv.write_text(rows_text)
        result = parse_diamond_taxonomy_output(tsv)
        assert "c1" in result


# ---------------------------------------------------------------------------
# build_gtdb_taxon_map / load_taxon_map
# ---------------------------------------------------------------------------


class TestTaxonMap:
    def test_build_gtdb_taxon_map(self, tmp_path):
        taxonomy_tsv = tmp_path / "bac120_taxonomy.tsv"
        taxonomy_tsv.write_text(
            "GB_GCA_000001405.29\t" + LIN_ECOLI + "\n"
            "RS_GCF_000006945.2\t" + LIN_KPNEU + "\n"
            "GB_GCA_000007545.1\t" + LIN_BSUB + "\n"
        )
        out_map = tmp_path / "taxon_map.tsv"
        result = build_gtdb_taxon_map(taxonomy_tsv, out_map)
        assert result.exists()

        loaded = load_taxon_map(out_map)
        assert "GCA_000001405.29" in loaded
        assert "GCF_000006945.2" in loaded
        assert loaded["GCA_000001405.29"] == LIN_ECOLI

    def test_load_taxon_map_comment_skip(self, tmp_path):
        map_file = tmp_path / "taxon_map.tsv"
        map_file.write_text("# header\nACC001\t" + LIN_ECOLI + "\n")
        result = load_taxon_map(map_file)
        assert "ACC001" in result
        assert "# header" not in result

    def test_load_taxon_map_empty(self, tmp_path):
        map_file = tmp_path / "taxon_map.tsv"
        map_file.write_text("")
        result = load_taxon_map(map_file)
        assert result == {}


# ---------------------------------------------------------------------------
# TaxResult properties
# ---------------------------------------------------------------------------


class TestTaxResultProperties:
    def test_display_classified(self):
        r = TaxResult(
            contig_id="c1", lineage=LIN_ECOLI, rank="species", taxon="s__Escherichia coli"
        )
        assert r.display == "species: s__Escherichia coli"

    def test_display_unclassified(self):
        r = TaxResult(contig_id="c1", lineage="", rank="unclassified", taxon="")
        assert r.display == "unclassified"

    def test_lineage_dict(self):
        r = TaxResult(
            contig_id="c1", lineage=LIN_ECOLI, rank="species", taxon="s__Escherichia coli"
        )
        d = r.lineage_dict
        assert d["domain"] == "d__Bacteria"
        assert d["genus"] == "g__Escherichia"
        assert d["species"] == "s__Escherichia coli"

    def test_lineage_dict_empty(self):
        r = TaxResult(contig_id="c1", lineage="", rank="unclassified", taxon="")
        assert r.lineage_dict == {}


# ---------------------------------------------------------------------------
# assign_taxonomy (mocked DIAMOND)
# ---------------------------------------------------------------------------


class TestAssignTaxonomy:
    def _write_diamond_tsv(self, path: Path) -> None:
        rows = [
            ["contig1", "ACC001", "95.0", "90.0", "1e-80", "500.0", LIN_ECOLI],
            ["contig1", "ACC002", "93.0", "88.0", "1e-78", "490.0", LIN_ECOLI],
            ["contig2", "ACC003", "91.0", "85.0", "1e-75", "470.0", LIN_KPNEU],
        ]
        with open(path, "w") as fh:
            for row in rows:
                fh.write("\t".join(str(c) for c in row) + "\n")

    def test_assign_taxonomy_returns_results(self, tmp_path):
        fasta = tmp_path / "contigs.fasta"
        fasta.write_text(">contig1\nACGT\n>contig2\nGGCC\n")
        db = tmp_path / "gtdb.dmnd"
        db.write_text("")

        diamond_tsv = tmp_path / "taxonomy" / "diamond_taxonomy.tsv"
        diamond_tsv.parent.mkdir(parents=True, exist_ok=True)
        self._write_diamond_tsv(diamond_tsv)

        with patch(
            "plasflow2.annotate.taxonomy.run_diamond_taxonomy",
            return_value=diamond_tsv,
        ):
            results = assign_taxonomy(
                fasta_path=fasta,
                taxonomy_db=db,
                work_dir=tmp_path / "taxonomy",
            )

        assert "contig1" in results
        assert "contig2" in results
        assert results["contig1"].rank == "species"
        assert results["contig1"].taxon == "s__Escherichia coli"

    def test_assign_taxonomy_with_taxon_map(self, tmp_path):
        fasta = tmp_path / "contigs.fasta"
        fasta.write_text(">contig1\nACGT\n")
        db = tmp_path / "gtdb.dmnd"
        db.write_text("")

        # Write a DIAMOND TSV with accession (no lineage in stitle)
        diamond_tsv = tmp_path / "taxonomy" / "diamond_taxonomy.tsv"
        diamond_tsv.parent.mkdir(parents=True, exist_ok=True)
        with open(diamond_tsv, "w") as fh:
            fh.write("contig1\tGCF_000001\t95.0\t90.0\t1e-80\t500.0\tprotein desc\n")

        # Write a taxon map
        taxon_map = tmp_path / "taxon_map.tsv"
        taxon_map.write_text("GCF_000001\t" + LIN_ECOLI + "\n")

        with patch(
            "plasflow2.annotate.taxonomy.run_diamond_taxonomy",
            return_value=diamond_tsv,
        ):
            results = assign_taxonomy(
                fasta_path=fasta,
                taxonomy_db=db,
                work_dir=tmp_path / "taxonomy",
                taxon_map_path=taxon_map,
            )

        assert "contig1" in results
        assert results["contig1"].taxon == "s__Escherichia coli"


# ---------------------------------------------------------------------------
# summarise_taxonomy
# ---------------------------------------------------------------------------


class TestSummariseTaxonomy:
    def test_basic_summary(self):
        taxonomy = {
            "c1": TaxResult("c1", LIN_ECOLI, "species", "s__Escherichia coli", 5, 0.9),
            "c2": TaxResult("c2", LIN_KPNEU, "genus", "g__Klebsiella", 5, 0.6),
            "c3": TaxResult("c3", "", "unclassified", "", 3, 0.0),
        }
        summary = summarise_taxonomy(taxonomy, total_contigs=5)
        assert summary.total_contigs == 5
        assert summary.classified == 2
        assert summary.unclassified == 3
        assert summary.rank_counts.get("species", 0) == 1
        assert summary.rank_counts.get("genus", 0) == 1

    def test_domain_counts(self):
        taxonomy = {
            "c1": TaxResult("c1", LIN_ECOLI, "domain", "d__Bacteria", 5, 0.9),
            "c2": TaxResult("c2", LIN_ARCH, "domain", "d__Archaea", 5, 0.9),
        }
        summary = summarise_taxonomy(taxonomy, total_contigs=2)
        assert summary.domain_counts.get("d__Bacteria", 0) == 1
        assert summary.domain_counts.get("d__Archaea", 0) == 1

    def test_empty_taxonomy(self):
        summary = summarise_taxonomy({}, total_contigs=10)
        assert summary.classified == 0
        assert summary.unclassified == 10
