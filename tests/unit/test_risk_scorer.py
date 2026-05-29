"""Unit tests for AMR risk scoring.

Scoring formula (capped at 10):
    Host taxonomy:   ESKAPE = 3, WHO priority = 2
    Mobility:        conjugative = 3, mobilizable = 2
    ARG burden:      ≥5 genes or ≥3 classes = 3 | 3-4 genes or 2 classes = 2 | 1-2 = 1
    Replicon:        broad-host-range (IncP/Q/W) = 2 | narrow = 1
    Source context:  clinical = 3 | wastewater/food = 2 | environmental = 1
"""

from plasflow2.annotate.args import ARGHit
from plasflow2.annotate.mobility import MobilityResult
from plasflow2.annotate.taxonomy import TaxResult
from plasflow2.risk.scorer import score_plasmid


def _mob(mobility_class: str, replicon: str = "unknown") -> MobilityResult:
    return MobilityResult(
        contig_id="test",
        mobility_class=mobility_class,
        replicon_type=replicon,
        relaxase_type="none",
        mpf_type="none",
    )


def _arg(gene: str = "blaTEM-1", drug_class: str = "beta-lactam") -> ARGHit:
    return ARGHit(
        contig_id="test",
        gene_name=gene,
        aro_accession="ARO:3000001",
        amr_family="TEM",
        drug_class=drug_class,
        resistance_mechanism="antibiotic inactivation",
        identity=99.0,
        coverage=100.0,
        evalue=1e-50,
    )


def _tax(genus: str = "", family: str = "", lineage: str = "") -> TaxResult:
    """Build a minimal TaxResult with GTDB-style lineage tokens."""
    if not lineage:
        parts = []
        if family:
            parts.append(f"f__{family}")
        if genus:
            parts.append(f"g__{genus}")
        lineage = ";".join(parts)
    return TaxResult(
        contig_id="test",
        taxon=f"g__{genus}" if genus else (f"f__{family}" if family else ""),
        rank="genus" if genus else ("family" if family else ""),
        lineage=lineage,
        agreement=1.0,
    )


# ---------------------------------------------------------------------------
# Baseline / mobility
# ---------------------------------------------------------------------------


def test_score_zero_for_non_mobilizable_no_args() -> None:
    result = score_plasmid("seq1", _mob("non-mobilizable"), [], "unspecified")
    assert result.score == 0


def test_score_conjugative_adds_three() -> None:
    result = score_plasmid("seq1", _mob("conjugative"), [], "unspecified")
    assert result.mobility_score == 3


def test_score_mobilizable_adds_two() -> None:
    result = score_plasmid("seq1", _mob("mobilizable"), [], "unspecified")
    assert result.mobility_score == 2


def test_score_capped_at_ten() -> None:
    args = [_arg(f"gene{i}", f"class{i}") for i in range(10)]
    result = score_plasmid(
        "seq1", _mob("conjugative", "IncP"), args, "clinical", taxonomy=_tax(genus="Klebsiella")
    )
    assert result.score == 10


def test_broad_host_range_incP() -> None:
    result = score_plasmid("seq1", _mob("non-mobilizable", "IncP"), [], "unspecified")
    assert result.replicon_score == 2


def test_narrow_replicon_adds_one() -> None:
    result = score_plasmid("seq1", _mob("non-mobilizable", "IncF"), [], "unspecified")
    assert result.replicon_score == 1


# ---------------------------------------------------------------------------
# Source context
# ---------------------------------------------------------------------------


def test_clinical_context_adds_three() -> None:
    result = score_plasmid("seq1", _mob("non-mobilizable"), [], "clinical")
    assert result.context_score == 3


def test_wastewater_context_adds_two() -> None:
    result = score_plasmid("seq1", _mob("non-mobilizable"), [], "wastewater")
    assert result.context_score == 2


def test_food_context_adds_two() -> None:
    result = score_plasmid("seq1", _mob("non-mobilizable"), [], "food")
    assert result.context_score == 2


def test_environmental_context_adds_one() -> None:
    result = score_plasmid("seq1", _mob("non-mobilizable"), [], "environmental")
    assert result.context_score == 1


def test_unspecified_context_adds_zero() -> None:
    result = score_plasmid("seq1", _mob("non-mobilizable"), [], "unspecified")
    assert result.context_score == 0


# ---------------------------------------------------------------------------
# Host taxonomy — ESKAPE / WHO priority
# ---------------------------------------------------------------------------


def test_eskape_klebsiella_adds_three() -> None:
    result = score_plasmid(
        "seq1", _mob("non-mobilizable"), [], "unspecified", taxonomy=_tax(genus="Klebsiella")
    )
    assert result.host_score == 3
    assert result.eskape_host is True
    assert result.eskape_genus == "Klebsiella"


def test_eskape_acinetobacter_adds_three() -> None:
    result = score_plasmid(
        "seq1", _mob("non-mobilizable"), [], "unspecified", taxonomy=_tax(genus="Acinetobacter")
    )
    assert result.host_score == 3


def test_eskape_via_enterobacteriaceae_family_adds_three() -> None:
    """Taxonomy resolved only to family level — still ESKAPE tier."""
    result = score_plasmid(
        "seq1",
        _mob("non-mobilizable"),
        [],
        "unspecified",
        taxonomy=_tax(family="Enterobacteriaceae"),
    )
    assert result.host_score == 3
    assert result.eskape_host is True


def test_who_priority_salmonella_adds_two() -> None:
    result = score_plasmid(
        "seq1", _mob("non-mobilizable"), [], "unspecified", taxonomy=_tax(genus="Salmonella")
    )
    assert result.host_score == 2
    assert result.eskape_host is True


def test_who_priority_mycobacterium_adds_two() -> None:
    result = score_plasmid(
        "seq1", _mob("non-mobilizable"), [], "unspecified", taxonomy=_tax(genus="Mycobacterium")
    )
    assert result.host_score == 2


def test_non_pathogen_host_adds_zero() -> None:
    result = score_plasmid(
        "seq1", _mob("non-mobilizable"), [], "unspecified", taxonomy=_tax(genus="Bacillus")
    )
    assert result.host_score == 0
    assert result.eskape_host is False


def test_no_taxonomy_adds_zero() -> None:
    result = score_plasmid("seq1", _mob("non-mobilizable"), [], "unspecified", taxonomy=None)
    assert result.host_score == 0


# ---------------------------------------------------------------------------
# ARG burden
# ---------------------------------------------------------------------------


def test_five_args_adds_three() -> None:
    args = [_arg(f"gene{i}", f"class{i}") for i in range(5)]
    result = score_plasmid("seq1", _mob("non-mobilizable"), args, "unspecified")
    assert result.arg_score == 3


def test_three_args_same_class_adds_two() -> None:
    args = [_arg(f"gene{i}", "beta-lactam") for i in range(3)]
    result = score_plasmid("seq1", _mob("non-mobilizable"), args, "unspecified")
    assert result.arg_score == 2


def test_one_arg_adds_one() -> None:
    result = score_plasmid("seq1", _mob("non-mobilizable"), [_arg()], "unspecified")
    assert result.arg_score == 1


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_evidence_is_populated() -> None:
    result = score_plasmid("seq1", _mob("conjugative", "IncP"), [_arg()], "wastewater")
    assert len(result.evidence) > 0


def test_no_mobility_result() -> None:
    """Should not crash when mobility is None."""
    result = score_plasmid("seq1", None, [], "unspecified")
    assert result.score == 0


def test_eskape_evidence_string_contains_plus_three() -> None:
    result = score_plasmid(
        "seq1", _mob("non-mobilizable"), [], "unspecified", taxonomy=_tax(genus="Pseudomonas")
    )
    assert any("+3" in e for e in result.evidence)
