"""Unit tests for AMR risk scoring.

Day 19 target: all tests pass.
"""

import pytest

from plasflow2.annotate.args import ARGHit
from plasflow2.annotate.mobility import MobilityResult
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
        amr_family="TEM",
        drug_class=drug_class,
        identity=99.0,
        coverage=100.0,
        evalue=1e-50,
    )


def test_score_zero_for_non_mobilizable_no_args() -> None:
    result = score_plasmid("seq1", _mob("non-mobilizable"), [], "unspecified")
    assert result.score == 0


def test_score_conjugative_adds_three() -> None:
    result = score_plasmid("seq1", _mob("conjugative"), [], "unspecified")
    assert result.mobility_score == 3


def test_score_capped_at_ten() -> None:
    args = [_arg(f"gene{i}", f"class{i}") for i in range(10)]
    result = score_plasmid("seq1", _mob("conjugative", "IncP"), args, "clinical")
    assert result.score == 10


def test_broad_host_range_incP() -> None:
    result = score_plasmid("seq1", _mob("non-mobilizable", "IncP"), [], "unspecified")
    assert result.replicon_score == 2


def test_clinical_context_adds_two() -> None:
    result = score_plasmid("seq1", _mob("non-mobilizable"), [], "clinical")
    assert result.context_score == 2


def test_evidence_is_populated() -> None:
    result = score_plasmid("seq1", _mob("conjugative", "IncP"), [_arg()], "wastewater")
    assert len(result.evidence) > 0


def test_no_mobility_result() -> None:
    """Should not crash when mobility is None."""
    result = score_plasmid("seq1", None, [], "unspecified")
    assert result.score == 0
