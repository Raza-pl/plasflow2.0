"""Unit tests for the HTML report generator (report/generator.py).

Tests build_report_data() and generate_report() with synthetic PipelineResult
data — no Jinja2, Plotly, or browser required for the data-builder tests.
"""

from __future__ import annotations

from pathlib import Path

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from plasflow2.annotate.args import ARGHit
from plasflow2.annotate.mobility import MobilityResult
from plasflow2.classify.predict import Prediction
from plasflow2.pipeline import ContigResult, PipelineResult
from plasflow2.report.generator import (
    PlasmidRow,
    _build_arg_chart,
    _build_pie_data,
    _build_risk_histogram,
    build_report_data,
    generate_report,
)
from plasflow2.risk.scorer import RiskScore

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_SEQ = "ACGT" * 500


def _record(name: str) -> SeqRecord:
    return SeqRecord(Seq(_SEQ), id=name, description="")


def _prediction(seq_id: str, label: str) -> Prediction:
    scores = {k: 0.0 for k in ("plasmid", "chromosome", "phage", "archaea")}
    scores[label] = 0.95
    return Prediction(sequence_id=seq_id, label=label, confidence=0.95, scores=scores)


def _arg(contig_id: str, drug_class: str = "carbapenem antibiotic") -> ARGHit:
    return ARGHit(
        contig_id=contig_id,
        gene_name="NDM-6",
        aro_accession="ARO:3002356",
        amr_family="NDM beta-lactamase",
        drug_class=drug_class,
        resistance_mechanism="antibiotic inactivation",
        identity=99.0,
        coverage=95.0,
        evalue=1e-120,
    )


def _mob(contig_id: str, mobility_class: str = "conjugative") -> MobilityResult:
    return MobilityResult(
        contig_id=contig_id,
        mobility_class=mobility_class,
        replicon_type="IncP-1alpha",
        relaxase_type="MOBP",
        mpf_type="MPF_T",
    )


def _risk(contig_id: str, score: int = 7) -> RiskScore:
    return RiskScore(
        contig_id=contig_id,
        score=score,
        evidence=["Conjugative (+3)", "IncP-1alpha (+2)"],
        mobility_score=3,
        arg_score=2,
        replicon_score=2,
        context_score=0,
    )


def _contig_result(name: str, num_args: int = 1, risk_score: int = 7) -> ContigResult:
    hits = [_arg(name) for _ in range(num_args)]
    return ContigResult(
        record=_record(name),
        prediction=_prediction(name, "plasmid"),
        arg_hits=hits,
        mobility=_mob(name),
        risk=_risk(name, risk_score),
    )


def _pipeline_result(num_plasmids: int = 2, num_chrom: int = 1) -> PipelineResult:
    plasmid_results = [_contig_result(f"p{i}", num_args=i % 3 + 1) for i in range(num_plasmids)]
    all_preds = [cr.prediction for cr in plasmid_results]
    all_preds += [_prediction(f"c{i}", "chromosome") for i in range(num_chrom)]
    return PipelineResult(
        input_fasta=Path("test.fasta"),
        all_predictions=all_preds,
        plasmid_results=plasmid_results,
    )


# ---------------------------------------------------------------------------
# _build_pie_data
# ---------------------------------------------------------------------------


def test_pie_data_structure() -> None:
    counts = {"plasmid": 10, "chromosome": 5, "phage": 2}
    result = _build_pie_data(counts)
    assert "data" in result
    assert "layout" in result
    assert result["data"][0]["type"] == "pie"


def test_pie_data_labels_values() -> None:
    counts = {"plasmid": 3, "chromosome": 7}
    result = _build_pie_data(counts)
    trace = result["data"][0]
    assert set(trace["labels"]) == {"plasmid", "chromosome"}
    assert sum(trace["values"]) == 10


# ---------------------------------------------------------------------------
# _build_arg_chart
# ---------------------------------------------------------------------------


def test_arg_chart_empty_hits() -> None:
    result = _build_arg_chart([])
    assert result["data"][0]["type"] == "bar"
    assert result["data"][0]["x"] == []


def test_arg_chart_counts_drug_classes() -> None:
    hits = [
        _arg("p1", "carbapenem antibiotic"),
        _arg("p1", "carbapenem antibiotic"),
        _arg("p2", "penicillin antibiotic"),
    ]
    result = _build_arg_chart(hits)
    y_labels = result["data"][0]["y"]
    x_counts = result["data"][0]["x"]
    label_to_count = dict(zip(y_labels, x_counts))
    assert label_to_count["carbapenem antibiotic"] == 2
    assert label_to_count["penicillin antibiotic"] == 1


def test_arg_chart_splits_semicolons() -> None:
    """Multi-class drug_class fields should be counted separately."""
    hits = [_arg("p1", "carbapenem antibiotic; cephalosporin antibiotic")]
    result = _build_arg_chart(hits)
    y_labels = result["data"][0]["y"]
    assert "carbapenem antibiotic" in y_labels
    assert "cephalosporin antibiotic" in y_labels


def test_arg_chart_ignores_unknown() -> None:
    hits = [_arg("p1", "unknown")]
    result = _build_arg_chart(hits)
    assert result["data"][0]["x"] == []


# ---------------------------------------------------------------------------
# _build_risk_histogram
# ---------------------------------------------------------------------------


def test_risk_histogram_structure() -> None:
    result = _build_risk_histogram([3, 5, 8, 8, 10])
    assert result["data"][0]["type"] == "bar"
    assert len(result["data"][0]["x"]) == 11  # 0..10


def test_risk_histogram_counts() -> None:
    result = _build_risk_histogram([3, 3, 8])
    y = result["data"][0]["y"]
    assert y[3] == 2  # two scores of 3
    assert y[8] == 1  # one score of 8


def test_risk_histogram_empty() -> None:
    result = _build_risk_histogram([])
    assert all(v == 0 for v in result["data"][0]["y"])


def test_risk_histogram_colors() -> None:
    result = _build_risk_histogram([])
    colors = result["data"][0]["marker"]["color"]
    assert colors[0] == "#27ae60"  # score 0 → green
    assert colors[4] == "#e67e22"  # score 4 → orange
    assert colors[7] == "#c0392b"  # score 7 → red
    assert colors[10] == "#c0392b"  # score 10 → red


# ---------------------------------------------------------------------------
# build_report_data
# ---------------------------------------------------------------------------


def test_build_report_data_keys() -> None:
    result = _pipeline_result()
    data = build_report_data(result, input_file="test.fasta")
    for key in (
        "input_file",
        "total",
        "num_plasmids",
        "total_args",
        "class_counts",
        "pie_data",
        "arg_data",
        "risk_data",
        "plasmid_rows",
    ):
        assert key in data, f"Missing key: {key}"


def test_build_report_data_counts() -> None:
    result = _pipeline_result(num_plasmids=3, num_chrom=2)
    data = build_report_data(result)
    assert data["total"] == 5
    assert data["num_plasmids"] == 3


def test_build_report_data_plasmid_rows_type() -> None:
    result = _pipeline_result(num_plasmids=2)
    data = build_report_data(result)
    assert all(isinstance(row, PlasmidRow) for row in data["plasmid_rows"])
    assert len(data["plasmid_rows"]) == 2


def test_build_report_data_row_fields() -> None:
    result = _pipeline_result(num_plasmids=1)
    data = build_report_data(result)
    row = data["plasmid_rows"][0]
    assert row.contig_id == "p0"
    assert row.mobility_class == "conjugative"
    assert row.replicon_type == "IncP-1alpha"
    assert row.risk_score == 7
    assert row.num_args >= 1


def test_build_report_data_no_mobility() -> None:
    """Contigs with mobility=None should show 'unknown' in the row."""
    cr = ContigResult(
        record=_record("p0"),
        prediction=_prediction("p0", "plasmid"),
        arg_hits=[],
        mobility=None,
        risk=_risk("p0", score=0),
    )
    result = PipelineResult(
        input_fasta=Path("x.fasta"),
        all_predictions=[_prediction("p0", "plasmid")],
        plasmid_results=[cr],
    )
    data = build_report_data(result)
    row = data["plasmid_rows"][0]
    assert row.mobility_class == "unknown"
    assert row.replicon_type == "unknown"


def test_build_report_data_drug_classes_joined() -> None:
    cr = ContigResult(
        record=_record("p0"),
        prediction=_prediction("p0", "plasmid"),
        arg_hits=[
            _arg("p0", "carbapenem antibiotic"),
            _arg("p0", "penicillin antibiotic"),
        ],
        mobility=_mob("p0"),
        risk=_risk("p0"),
    )
    result = PipelineResult(
        input_fasta=Path("x.fasta"),
        all_predictions=[_prediction("p0", "plasmid")],
        plasmid_results=[cr],
    )
    data = build_report_data(result)
    dc = data["plasmid_rows"][0].drug_classes
    assert "carbapenem antibiotic" in dc
    assert "penicillin antibiotic" in dc


def test_build_report_data_no_args_dash() -> None:
    """Contigs with no ARGs should show '—' in drug_classes."""
    cr = ContigResult(
        record=_record("p0"),
        prediction=_prediction("p0", "plasmid"),
        arg_hits=[],
        mobility=_mob("p0"),
        risk=_risk("p0", score=3),
    )
    result = PipelineResult(
        input_fasta=Path("x.fasta"),
        all_predictions=[_prediction("p0", "plasmid")],
        plasmid_results=[cr],
    )
    data = build_report_data(result)
    assert data["plasmid_rows"][0].drug_classes == "—"


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


def test_generate_report_creates_file(tmp_path: Path) -> None:
    result = _pipeline_result(num_plasmids=2)
    data = build_report_data(result, input_file="test.fasta")
    out = tmp_path / "report.html"
    path = generate_report(data, out)
    assert path == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_generate_report_contains_plotly(tmp_path: Path) -> None:
    result = _pipeline_result(num_plasmids=1)
    data = build_report_data(result)
    out = generate_report(data, tmp_path / "report.html")
    html = out.read_text()
    assert "plotly" in html.lower()
    assert "pie-chart" in html


def test_generate_report_contains_contig_id(tmp_path: Path) -> None:
    result = _pipeline_result(num_plasmids=1)
    data = build_report_data(result)
    out = generate_report(data, tmp_path / "report.html")
    html = out.read_text()
    assert "p0" in html


def test_generate_report_creates_parent_dir(tmp_path: Path) -> None:
    result = _pipeline_result(num_plasmids=1)
    data = build_report_data(result)
    out = tmp_path / "nested" / "dir" / "report.html"
    generate_report(data, out)
    assert out.exists()
