"""Unit tests for the end-to-end pipeline (pipeline.py).

All external calls (predict, annotate_contigs, run_mob_typer) are mocked so
these tests run without DIAMOND, mob_typer, or trained model weights.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from plasflow2.annotate.args import ARGHit
from plasflow2.annotate.mobility import MobilityResult
from plasflow2.classify.predict import Prediction
from plasflow2.pipeline import ContigResult, PipelineResult, run_pipeline
from plasflow2.risk.scorer import RiskScore

# ---------------------------------------------------------------------------
# Helpers for synthetic data
# ---------------------------------------------------------------------------

_SEQ = "ACGT" * 500  # 2000 bp — passes min_length=1000


def _record(name: str, seq: str = _SEQ) -> SeqRecord:
    return SeqRecord(Seq(seq), id=name, description="")


def _prediction(seq_id: str, label: str, confidence: float = 0.95) -> Prediction:
    scores = {"plasmid": 0.0, "chromosome": 0.0, "phage": 0.0, "archaea": 0.0}
    if label in scores:
        scores[label] = confidence
    return Prediction(sequence_id=seq_id, label=label, confidence=confidence, scores=scores)


def _arg_hit(contig_id: str) -> ARGHit:
    return ARGHit(
        contig_id=contig_id,
        gene_name="NDM-6",
        aro_accession="ARO:3002356",
        amr_family="NDM beta-lactamase",
        drug_class="carbapenem antibiotic",
        resistance_mechanism="antibiotic inactivation",
        identity=99.5,
        coverage=95.0,
        evalue=1e-120,
    )


def _mob_result(contig_id: str, mobility_class: str = "conjugative") -> MobilityResult:
    return MobilityResult(
        contig_id=contig_id,
        mobility_class=mobility_class,
        replicon_type="IncP-1alpha",
        relaxase_type="MOBP",
        mpf_type="MPF_T",
    )


def _risk(contig_id: str, score: int = 5) -> RiskScore:
    return RiskScore(contig_id=contig_id, score=score, evidence=["mock evidence"])


# ---------------------------------------------------------------------------
# Shared mock factory
# ---------------------------------------------------------------------------


def _mock_pipeline(
    tmp_path: Path,
    *,
    fasta_records: list[SeqRecord],
    predictions: list[Prediction],
    arg_hits: list[ARGHit] | None = None,
    mob_results: list[MobilityResult] | None = None,
    skip_mobility: bool = False,
) -> PipelineResult:
    """Run run_pipeline() with all external I/O mocked."""
    fasta = tmp_path / "contigs.fasta"
    # Write a minimal FASTA so FileNotFoundError checks pass
    fasta.write_text("".join(f">{r.id}\n{r.seq}\n" for r in fasta_records))

    model = tmp_path / "mlp_v2.pt"
    model.write_text("mock")
    card_db = tmp_path / "card.dmnd"
    card_db.write_text("mock")
    aro_index = tmp_path / "aro_index.tsv"
    aro_index.write_text("mock")

    with (
        patch("plasflow2.pipeline.load_fasta", return_value=fasta_records),
        patch("plasflow2.pipeline.predict", return_value=predictions),
        patch("plasflow2.pipeline.write_fasta"),
        patch("plasflow2.pipeline.annotate_contigs", return_value=arg_hits or []),
        patch("plasflow2.pipeline.run_mob_typer", return_value=tmp_path / "mob.txt"),
        patch("plasflow2.pipeline.parse_mob_results", return_value=mob_results or []),
    ):
        return run_pipeline(
            fasta_path=fasta,
            model_path=model,
            card_db=card_db,
            aro_index=aro_index,
            work_dir=tmp_path / "work",
            skip_mobility=skip_mobility,
        )


# ---------------------------------------------------------------------------
# PipelineResult.__post_init__ (no mocking needed)
# ---------------------------------------------------------------------------


def test_pipeline_result_counts_class_labels() -> None:
    preds = [
        _prediction("c1", "plasmid"),
        _prediction("c2", "chromosome"),
        _prediction("c3", "plasmid"),
        _prediction("c4", "phage"),
    ]
    result = PipelineResult(
        input_fasta=Path("x.fasta"),
        all_predictions=preds,
        plasmid_results=[],
    )
    assert result.class_counts["plasmid"] == 2
    assert result.class_counts["chromosome"] == 1
    assert result.class_counts["phage"] == 1
    assert result.total_sequences == 4
    assert result.total_plasmids == 0


def test_pipeline_result_total_args() -> None:
    cr = ContigResult(
        record=_record("p1"),
        prediction=_prediction("p1", "plasmid"),
        arg_hits=[_arg_hit("p1"), _arg_hit("p1")],
        mobility=None,
        risk=_risk("p1"),
    )
    result = PipelineResult(
        input_fasta=Path("x.fasta"),
        all_predictions=[_prediction("p1", "plasmid")],
        plasmid_results=[cr],
    )
    assert result.total_args == 2


# ---------------------------------------------------------------------------
# run_pipeline — basic flow
# ---------------------------------------------------------------------------


def test_run_pipeline_empty_fasta_returns_empty(tmp_path: Path) -> None:
    result = _mock_pipeline(tmp_path, fasta_records=[], predictions=[])
    assert result.total_sequences == 0
    assert result.plasmid_results == []


def test_run_pipeline_no_plasmids_returns_empty_plasmid_list(tmp_path: Path) -> None:
    records = [_record("c1"), _record("c2")]
    preds = [_prediction("c1", "chromosome"), _prediction("c2", "phage")]
    result = _mock_pipeline(tmp_path, fasta_records=records, predictions=preds)
    assert result.total_plasmids == 0
    assert result.plasmid_results == []
    assert result.total_sequences == 2


def test_run_pipeline_plasmid_count(tmp_path: Path) -> None:
    records = [_record("p1"), _record("c1"), _record("p2")]
    preds = [
        _prediction("p1", "plasmid"),
        _prediction("c1", "chromosome"),
        _prediction("p2", "plasmid"),
    ]
    result = _mock_pipeline(tmp_path, fasta_records=records, predictions=preds, skip_mobility=True)
    assert result.total_plasmids == 2


def test_run_pipeline_arg_hits_grouped_by_contig(tmp_path: Path) -> None:
    records = [_record("p1")]
    preds = [_prediction("p1", "plasmid")]
    hits = [_arg_hit("p1"), _arg_hit("p1")]
    result = _mock_pipeline(
        tmp_path,
        fasta_records=records,
        predictions=preds,
        arg_hits=hits,
        skip_mobility=True,
    )
    assert len(result.plasmid_results[0].arg_hits) == 2
    assert result.total_args == 2


def test_run_pipeline_mobility_attached(tmp_path: Path) -> None:
    records = [_record("p1")]
    preds = [_prediction("p1", "plasmid")]
    mob = [_mob_result("p1", "conjugative")]
    result = _mock_pipeline(
        tmp_path,
        fasta_records=records,
        predictions=preds,
        mob_results=mob,
    )
    assert result.plasmid_results[0].mobility is not None
    assert result.plasmid_results[0].mobility.mobility_class == "conjugative"


def test_run_pipeline_skip_mobility_sets_none(tmp_path: Path) -> None:
    records = [_record("p1")]
    preds = [_prediction("p1", "plasmid")]
    result = _mock_pipeline(
        tmp_path,
        fasta_records=records,
        predictions=preds,
        skip_mobility=True,
    )
    assert result.plasmid_results[0].mobility is None


def test_run_pipeline_risk_score_present(tmp_path: Path) -> None:
    records = [_record("p1")]
    preds = [_prediction("p1", "plasmid")]
    result = _mock_pipeline(
        tmp_path,
        fasta_records=records,
        predictions=preds,
        skip_mobility=True,
    )
    assert isinstance(result.plasmid_results[0].risk, RiskScore)


# ---------------------------------------------------------------------------
# run_pipeline — FileNotFoundError checks
# ---------------------------------------------------------------------------


def test_run_pipeline_missing_fasta_raises(tmp_path: Path) -> None:
    model = tmp_path / "mlp_v2.pt"
    model.write_text("mock")
    card_db = tmp_path / "card.dmnd"
    card_db.write_text("mock")
    aro_index = tmp_path / "aro_index.tsv"
    aro_index.write_text("mock")
    with pytest.raises(FileNotFoundError, match="fasta_path"):
        run_pipeline(
            fasta_path=tmp_path / "nonexistent.fasta",
            model_path=model,
            card_db=card_db,
            aro_index=aro_index,
            work_dir=tmp_path / "work",
        )


def test_run_pipeline_missing_model_raises(tmp_path: Path) -> None:
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(">c1\nACGT\n")
    card_db = tmp_path / "card.dmnd"
    card_db.write_text("mock")
    aro_index = tmp_path / "aro_index.tsv"
    aro_index.write_text("mock")
    with pytest.raises(FileNotFoundError, match="model_path"):
        run_pipeline(
            fasta_path=fasta,
            model_path=tmp_path / "missing.pt",
            card_db=card_db,
            aro_index=aro_index,
            work_dir=tmp_path / "work",
        )


# ---------------------------------------------------------------------------
# run_pipeline — mob_typer failure is graceful
# ---------------------------------------------------------------------------


def test_run_pipeline_mob_typer_failure_is_graceful(tmp_path: Path) -> None:
    """If mob_typer raises RuntimeError, pipeline continues with mobility=None."""
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(f">p1\n{_SEQ}\n")
    model = tmp_path / "mlp_v2.pt"
    model.write_text("mock")
    card_db = tmp_path / "card.dmnd"
    card_db.write_text("mock")
    aro_index = tmp_path / "aro_index.tsv"
    aro_index.write_text("mock")

    with (
        patch("plasflow2.pipeline.load_fasta", return_value=[_record("p1")]),
        patch("plasflow2.pipeline.predict", return_value=[_prediction("p1", "plasmid")]),
        patch("plasflow2.pipeline.write_fasta"),
        patch("plasflow2.pipeline.annotate_contigs", return_value=[]),
        patch("plasflow2.pipeline.run_mob_typer", side_effect=RuntimeError("mob_typer not found")),
    ):
        result = run_pipeline(
            fasta_path=fasta,
            model_path=model,
            card_db=card_db,
            aro_index=aro_index,
            work_dir=tmp_path / "work",
        )

    assert result.total_plasmids == 1
    assert result.plasmid_results[0].mobility is None
