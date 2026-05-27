"""Integration tests for the full PlasFlow v2 pipeline.

Days 25-28: These tests exercise real Python logic end-to-end, with external
binaries (DIAMOND, mob_typer, pyrodigal) mocked at the subprocess/function level.
They verify correct data flow across all stages and validate actual output files.

Test categories:
  1. run_pipeline() — core Python chain (classify → annotate → risk → report)
  2. CLI `plasflow2 run` — CliRunner end-to-end with file output validation
  3. CLI `plasflow2 classify` — standalone classify subcommand
  4. CLI `plasflow2 annotate` — standalone annotate subcommand
  5. CLI `plasflow2 report` — roundtrip from existing JSON/TSV
  6. Edge cases: empty FASTA, no plasmids, mob_typer failure, high-risk scenario
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from click.testing import CliRunner
from plasflow2.annotate.args import ARGHit
from plasflow2.annotate.mobility import MobilityResult
from plasflow2.classify.predict import Prediction
from plasflow2.cli import main
from plasflow2.pipeline import ContigResult, PipelineResult, run_pipeline
from plasflow2.risk.scorer import RiskScore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 2000 bp — passes the default min_length=1000 filter
_SEQ = "ACGT" * 500
_SHORT_SEQ = "ACGT" * 100  # 400 bp — filtered out by min_length=1000


# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------


def _record(name: str, seq: str = _SEQ) -> SeqRecord:
    return SeqRecord(Seq(seq), id=name, description="")


def _prediction(seq_id: str, label: str, confidence: float = 0.95) -> Prediction:
    scores = {"plasmid": 0.0, "chromosome": 0.0, "phage": 0.0, "archaea": 0.0}
    if label in scores:
        scores[label] = confidence
    return Prediction(sequence_id=seq_id, label=label, confidence=confidence, scores=scores)


def _arg_hit(
    contig_id: str,
    gene_name: str = "NDM-6",
    drug_class: str = "carbapenem antibiotic",
) -> ARGHit:
    return ARGHit(
        contig_id=contig_id,
        gene_name=gene_name,
        aro_accession="ARO:3002356",
        amr_family="NDM beta-lactamase",
        drug_class=drug_class,
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


def _write_fasta(path: Path, records: dict[str, str]) -> None:
    """Write a minimal FASTA file from {name: seq} dict."""
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in records.items()))


def _mock_pipeline_run(
    tmp_path: Path,
    *,
    records: list[SeqRecord],
    predictions: list[Prediction],
    arg_hits: list[ARGHit] | None = None,
    mob_results: list[MobilityResult] | None = None,
    skip_mobility: bool = False,
) -> PipelineResult:
    """Run run_pipeline() with all external I/O mocked."""
    fasta = tmp_path / "contigs.fasta"
    _write_fasta(fasta, {r.id: str(r.seq) for r in records})

    model = tmp_path / "mlp_v2.pt"
    model.write_text("mock")
    card_db = tmp_path / "card.dmnd"
    card_db.write_text("mock")
    aro_index = tmp_path / "aro_index.tsv"
    aro_index.write_text("mock")

    with (
        patch("plasflow2.pipeline.load_fasta", return_value=records),
        patch("plasflow2.pipeline.predict", return_value=predictions),
        patch("plasflow2.pipeline.write_fasta"),
        patch("plasflow2.pipeline.annotate_contigs", return_value=arg_hits or []),
        patch("plasflow2.pipeline.run_mob_typer", return_value=tmp_path / "mob_results.txt"),
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
# 1. run_pipeline() — data flow integration
# ---------------------------------------------------------------------------


def test_pipeline_plasmid_and_chromosome_mix(tmp_path: Path) -> None:
    """Mixed input: plasmids get annotated + risk-scored; chromosomes are counted only."""
    records = [_record("p1"), _record("p2"), _record("c1"), _record("c2")]
    preds = [
        _prediction("p1", "plasmid"),
        _prediction("p2", "plasmid"),
        _prediction("c1", "chromosome"),
        _prediction("c2", "chromosome"),
    ]
    arg_hits = [_arg_hit("p1"), _arg_hit("p1"), _arg_hit("p2")]
    mob = [_mob_result("p1", "conjugative"), _mob_result("p2", "mobilizable")]

    result = _mock_pipeline_run(
        tmp_path,
        records=records,
        predictions=preds,
        arg_hits=arg_hits,
        mob_results=mob,
    )

    assert result.total_sequences == 4
    assert result.total_plasmids == 2
    assert result.class_counts["plasmid"] == 2
    assert result.class_counts["chromosome"] == 2
    assert result.total_args == 3

    # Per-contig data
    contig_ids = {cr.record.id for cr in result.plasmid_results}
    assert contig_ids == {"p1", "p2"}

    p1 = next(cr for cr in result.plasmid_results if cr.record.id == "p1")
    assert len(p1.arg_hits) == 2
    assert p1.mobility is not None
    assert p1.mobility.mobility_class == "conjugative"
    assert p1.risk is not None
    assert p1.risk.score > 0


def test_pipeline_no_plasmids_returns_empty_annotation(tmp_path: Path) -> None:
    """When no plasmids are detected, plasmid_results is empty and total_args is 0."""
    records = [_record("c1"), _record("c2")]
    preds = [_prediction("c1", "chromosome"), _prediction("c2", "phage")]

    result = _mock_pipeline_run(tmp_path, records=records, predictions=preds, skip_mobility=True)

    assert result.total_plasmids == 0
    assert result.total_args == 0
    assert result.plasmid_results == []
    assert result.class_counts.get("chromosome", 0) == 1
    assert result.class_counts.get("phage", 0) == 1


def test_pipeline_empty_fasta_returns_zero_counts(tmp_path: Path) -> None:
    """Empty input FASTA → all counts zero, empty results."""
    result = _mock_pipeline_run(tmp_path, records=[], predictions=[])
    assert result.total_sequences == 0
    assert result.total_plasmids == 0
    assert result.total_args == 0
    assert result.plasmid_results == []


def test_pipeline_skip_mobility_sets_mobility_none(tmp_path: Path) -> None:
    """With skip_mobility=True, all ContigResult.mobility fields are None."""
    records = [_record("p1"), _record("p2")]
    preds = [_prediction("p1", "plasmid"), _prediction("p2", "plasmid")]

    result = _mock_pipeline_run(tmp_path, records=records, predictions=preds, skip_mobility=True)

    assert result.total_plasmids == 2
    for cr in result.plasmid_results:
        assert cr.mobility is None


def test_pipeline_high_risk_scenario(tmp_path: Path) -> None:
    """Conjugative plasmid + 6 ARGs across 3 drug classes → risk score ≥ 7."""
    records = [_record("p_highrisk")]
    preds = [_prediction("p_highrisk", "plasmid")]
    arg_hits = [
        _arg_hit("p_highrisk", "NDM-6", "carbapenem antibiotic"),
        _arg_hit("p_highrisk", "NDM-1", "carbapenem antibiotic"),
        _arg_hit("p_highrisk", "TEM-1", "penicillin antibiotic"),
        _arg_hit("p_highrisk", "CTX-M-15", "cephalosporin antibiotic"),
        _arg_hit("p_highrisk", "CTX-M-14", "cephalosporin antibiotic"),
        _arg_hit("p_highrisk", "MCR-1", "polymyxin antibiotic"),
    ]
    mob = [_mob_result("p_highrisk", "conjugative")]

    result = _mock_pipeline_run(
        tmp_path,
        records=records,
        predictions=preds,
        arg_hits=arg_hits,
        mob_results=mob,
    )

    cr = result.plasmid_results[0]
    assert cr.risk.score >= 7, f"Expected high risk, got {cr.risk.score}"
    assert cr.risk.mobility_score > 0
    assert cr.risk.arg_score > 0


def test_pipeline_mob_typer_failure_continues_gracefully(tmp_path: Path) -> None:
    """RuntimeError from run_mob_typer → pipeline continues with mobility=None."""
    fasta = tmp_path / "contigs.fasta"
    _write_fasta(fasta, {"p1": _SEQ})
    model = tmp_path / "mlp.pt"
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
    assert isinstance(result.plasmid_results[0].risk, RiskScore)


def test_pipeline_arg_grouping_by_contig(tmp_path: Path) -> None:
    """ARG hits are correctly assigned to their respective contigs."""
    records = [_record("p1"), _record("p2")]
    preds = [_prediction("p1", "plasmid"), _prediction("p2", "plasmid")]
    arg_hits = [
        _arg_hit("p1", "NDM-6"),
        _arg_hit("p1", "TEM-1"),
        _arg_hit("p2", "CTX-M-15"),
    ]

    result = _mock_pipeline_run(
        tmp_path, records=records, predictions=preds, arg_hits=arg_hits, skip_mobility=True
    )

    p1 = next(cr for cr in result.plasmid_results if cr.record.id == "p1")
    p2 = next(cr for cr in result.plasmid_results if cr.record.id == "p2")
    assert len(p1.arg_hits) == 2
    assert len(p2.arg_hits) == 1
    assert {h.gene_name for h in p1.arg_hits} == {"NDM-6", "TEM-1"}


# ---------------------------------------------------------------------------
# 2. CLI `plasflow2 run` — end-to-end with file output validation
# ---------------------------------------------------------------------------


def test_cli_run_writes_all_output_files(tmp_path: Path) -> None:
    """CLI `run` produces predictions.tsv, annotations.json, and report.html."""
    fasta = tmp_path / "contigs.fasta"
    _write_fasta(fasta, {"p1": _SEQ, "c1": _SEQ})
    model = tmp_path / "mlp.pt"
    model.write_text("mock")
    card_db = tmp_path / "card.dmnd"
    card_db.write_text("mock")
    aro_index = tmp_path / "aro_index.tsv"
    aro_index.write_text("mock")
    out = tmp_path / "output"

    preds = [_prediction("p1", "plasmid"), _prediction("c1", "chromosome")]
    p1_cr = ContigResult(
        record=_record("p1"),
        prediction=_prediction("p1", "plasmid"),
        arg_hits=[_arg_hit("p1")],
        mobility=_mob_result("p1"),
        risk=RiskScore(contig_id="p1", score=7, evidence=["Conjugative (+3)"]),
    )
    pipeline_result = PipelineResult(
        input_fasta=fasta,
        all_predictions=preds,
        plasmid_results=[p1_cr],
    )

    runner = CliRunner()
    with (
        patch("plasflow2.cli.run_pipeline", return_value=pipeline_result),
        patch(
            "plasflow2.cli.load_fasta",
            return_value=[_record("p1"), _record("c1")],
        ),
        patch("plasflow2.cli.write_fasta"),
    ):
        result = runner.invoke(
            main,
            [
                "run",
                "--input",
                str(fasta),
                "--output",
                str(out),
                "--model",
                str(model),
                "--card-db",
                str(card_db),
                "--aro-index",
                str(aro_index),
                "--skip-mobility",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Done." in result.output

    # predictions.tsv
    tsv = out / "predictions.tsv"
    assert tsv.exists()
    rows = list(csv.DictReader(tsv.open(), delimiter="\t"))
    assert len(rows) == 2
    labels = {r["sequence_id"]: r["label"] for r in rows}
    assert labels["p1"] == "plasmid"
    assert labels["c1"] == "chromosome"

    # annotations.json
    ann = out / "annotations.json"
    assert ann.exists()
    data = json.loads(ann.read_text())
    assert len(data) == 1
    assert data[0]["contig_id"] == "p1"
    assert len(data[0]["arg_hits"]) == 1
    assert data[0]["risk"]["score"] == 7

    # report.html
    html = out / "report.html"
    assert html.exists()
    html_text = html.read_text()
    assert "PlasFlow" in html_text
    assert "p1" in html_text


def test_cli_run_predictions_tsv_columns(tmp_path: Path) -> None:
    """predictions.tsv has the correct header columns in the correct order."""
    fasta = tmp_path / "contigs.fasta"
    _write_fasta(fasta, {"s1": _SEQ})
    model = tmp_path / "mlp.pt"
    model.write_text("mock")
    card_db = tmp_path / "card.dmnd"
    card_db.write_text("mock")
    aro_index = tmp_path / "aro_index.tsv"
    aro_index.write_text("mock")
    out = tmp_path / "output"

    pipeline_result = PipelineResult(
        input_fasta=fasta,
        all_predictions=[_prediction("s1", "plasmid")],
        plasmid_results=[
            ContigResult(
                record=_record("s1"),
                prediction=_prediction("s1", "plasmid"),
                arg_hits=[],
                mobility=None,
                risk=RiskScore(contig_id="s1", score=0, evidence=[]),
            )
        ],
    )

    runner = CliRunner()
    with (
        patch("plasflow2.cli.run_pipeline", return_value=pipeline_result),
        patch("plasflow2.cli.load_fasta", return_value=[_record("s1")]),
        patch("plasflow2.cli.write_fasta"),
    ):
        result = runner.invoke(
            main,
            [
                "run",
                "--input",
                str(fasta),
                "--output",
                str(out),
                "--model",
                str(model),
                "--card-db",
                str(card_db),
                "--aro-index",
                str(aro_index),
                "--skip-mobility",
            ],
        )

    assert result.exit_code == 0, result.output
    with open(out / "predictions.tsv") as fh:
        header = fh.readline().strip().split("\t")
    assert header == [
        "sequence_id",
        "label",
        "confidence",
        "plasmid",
        "chromosome",
        "phage",
        "archaea",
    ]


def test_cli_run_annotations_json_structure(tmp_path: Path) -> None:
    """annotations.json contains the expected nested structure per plasmid."""
    fasta = tmp_path / "contigs.fasta"
    _write_fasta(fasta, {"p1": _SEQ})
    model = tmp_path / "mlp.pt"
    model.write_text("mock")
    card_db = tmp_path / "card.dmnd"
    card_db.write_text("mock")
    aro_index = tmp_path / "aro_index.tsv"
    aro_index.write_text("mock")
    out = tmp_path / "output"

    pipeline_result = PipelineResult(
        input_fasta=fasta,
        all_predictions=[_prediction("p1", "plasmid")],
        plasmid_results=[
            ContigResult(
                record=_record("p1"),
                prediction=_prediction("p1", "plasmid"),
                arg_hits=[_arg_hit("p1", "NDM-6", "carbapenem antibiotic")],
                mobility=_mob_result("p1", "conjugative"),
                risk=RiskScore(
                    contig_id="p1",
                    score=8,
                    evidence=["Conjugative (+3)", "IncP-1alpha (+2)", "NDM-6 (+3)"],
                    mobility_score=3,
                    arg_score=3,
                    replicon_score=2,
                    context_score=0,
                ),
            )
        ],
    )

    runner = CliRunner()
    with (
        patch("plasflow2.cli.run_pipeline", return_value=pipeline_result),
        patch("plasflow2.cli.load_fasta", return_value=[_record("p1")]),
        patch("plasflow2.cli.write_fasta"),
    ):
        runner.invoke(
            main,
            [
                "run",
                "--input",
                str(fasta),
                "--output",
                str(out),
                "--model",
                str(model),
                "--card-db",
                str(card_db),
                "--aro-index",
                str(aro_index),
                "--skip-mobility",
            ],
        )

    data = json.loads((out / "annotations.json").read_text())
    assert len(data) == 1
    rec = data[0]

    # Top-level keys
    for key in ("contig_id", "length", "classification", "mobility", "arg_hits", "risk"):
        assert key in rec, f"Missing key: {key}"

    assert rec["contig_id"] == "p1"
    assert rec["length"] == 2000
    assert rec["classification"]["label"] == "plasmid"
    assert rec["classification"]["confidence"] == pytest.approx(0.95)

    # Mobility block
    mob = rec["mobility"]
    assert mob is not None
    assert mob["mobility_class"] == "conjugative"
    assert mob["replicon_type"] == "IncP-1alpha"

    # ARG hits
    assert len(rec["arg_hits"]) == 1
    hit = rec["arg_hits"][0]
    assert hit["gene_name"] == "NDM-6"
    assert hit["drug_class"] == "carbapenem antibiotic"

    # Risk block
    risk = rec["risk"]
    assert risk["score"] == 8
    assert risk["mobility_score"] == 3
    assert risk["arg_score"] == 3
    assert risk["replicon_score"] == 2


# ---------------------------------------------------------------------------
# 3. CLI `plasflow2 classify` — standalone
# ---------------------------------------------------------------------------


def test_cli_classify_writes_tsv_with_correct_rows(tmp_path: Path) -> None:
    """classify subcommand writes one row per input sequence."""
    fasta = tmp_path / "input.fasta"
    _write_fasta(fasta, {"p1": _SEQ, "c1": _SEQ, "ph1": _SEQ})
    model = tmp_path / "mlp.pt"
    model.write_text("mock")
    out_tsv = tmp_path / "preds.tsv"

    preds = [
        _prediction("p1", "plasmid"),
        _prediction("c1", "chromosome"),
        _prediction("ph1", "phage"),
    ]

    runner = CliRunner()
    with (
        patch(
            "plasflow2.cli.load_fasta",
            return_value=[_record("p1"), _record("c1"), _record("ph1")],
        ),
        patch("plasflow2.cli.predict", return_value=preds),
    ):
        result = runner.invoke(
            main,
            ["classify", "--input", str(fasta), "--output", str(out_tsv), "--model", str(model)],
        )

    assert result.exit_code == 0, result.output
    assert out_tsv.exists()

    rows = list(csv.DictReader(out_tsv.open(), delimiter="\t"))
    assert len(rows) == 3
    labels = {r["sequence_id"]: r["label"] for r in rows}
    assert labels == {"p1": "plasmid", "c1": "chromosome", "ph1": "phage"}


def test_cli_classify_confidence_values_in_tsv(tmp_path: Path) -> None:
    """classify writes correctly formatted confidence and score columns."""
    fasta = tmp_path / "input.fasta"
    _write_fasta(fasta, {"s1": _SEQ})
    model = tmp_path / "mlp.pt"
    model.write_text("mock")
    out_tsv = tmp_path / "preds.tsv"

    pred = _prediction("s1", "plasmid", confidence=0.9876)

    runner = CliRunner()
    with (
        patch("plasflow2.cli.load_fasta", return_value=[_record("s1")]),
        patch("plasflow2.cli.predict", return_value=[pred]),
    ):
        runner.invoke(
            main,
            ["classify", "--input", str(fasta), "--output", str(out_tsv), "--model", str(model)],
        )

    rows = list(csv.DictReader(out_tsv.open(), delimiter="\t"))
    assert len(rows) == 1
    row = rows[0]
    assert float(row["confidence"]) == pytest.approx(0.9876, abs=1e-3)
    # plasmid score should be the dominant value
    assert float(row["plasmid"]) == pytest.approx(0.9876, abs=1e-3)
    # other scores should be ~0
    assert float(row["chromosome"]) == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 4. CLI `plasflow2 annotate` — standalone
# ---------------------------------------------------------------------------


def test_cli_annotate_writes_annotations_json(tmp_path: Path) -> None:
    """annotate subcommand produces annotations.json with ARG + mobility data."""
    fasta = tmp_path / "plasmids.fasta"
    _write_fasta(fasta, {"p1": _SEQ})
    card_db = tmp_path / "card.dmnd"
    card_db.write_text("mock")
    aro_index = tmp_path / "aro_index.tsv"
    aro_index.write_text("mock")
    out_dir = tmp_path / "ann_out"

    arg_hits = [_arg_hit("p1", "NDM-6", "carbapenem antibiotic")]
    mob_results = [_mob_result("p1", "conjugative")]

    runner = CliRunner()
    with (
        patch("plasflow2.cli.annotate_contigs", return_value=arg_hits),
        patch("plasflow2.cli.annotate_mobility", return_value=mob_results),
    ):
        result = runner.invoke(
            main,
            [
                "annotate",
                "--input",
                str(fasta),
                "--output",
                str(out_dir),
                "--card-db",
                str(card_db),
                "--aro-index",
                str(aro_index),
            ],
        )

    assert result.exit_code == 0, result.output
    ann_json = out_dir / "annotations.json"
    assert ann_json.exists()

    data = json.loads(ann_json.read_text())
    assert len(data) == 1
    assert data[0]["contig_id"] == "p1"
    assert len(data[0]["arg_hits"]) == 1
    assert data[0]["arg_hits"][0]["gene_name"] == "NDM-6"
    assert data[0]["mobility"]["mobility_class"] == "conjugative"


def test_cli_annotate_skip_mobility_not_called(tmp_path: Path) -> None:
    """annotate with --skip-mobility never calls annotate_mobility."""
    fasta = tmp_path / "plasmids.fasta"
    _write_fasta(fasta, {"p1": _SEQ})
    card_db = tmp_path / "card.dmnd"
    card_db.write_text("mock")
    aro_index = tmp_path / "aro_index.tsv"
    aro_index.write_text("mock")

    runner = CliRunner()
    with (
        patch("plasflow2.cli.annotate_contigs", return_value=[]),
        patch("plasflow2.cli.annotate_mobility") as mock_mob,
    ):
        result = runner.invoke(
            main,
            [
                "annotate",
                "--input",
                str(fasta),
                "--output",
                str(tmp_path / "out"),
                "--card-db",
                str(card_db),
                "--aro-index",
                str(aro_index),
                "--skip-mobility",
            ],
        )
    assert result.exit_code == 0, result.output
    mock_mob.assert_not_called()


# ---------------------------------------------------------------------------
# 5. CLI `plasflow2 report` — roundtrip from existing files
# ---------------------------------------------------------------------------


def _write_test_files(tmp_path: Path) -> tuple[Path, Path]:
    """Write minimal annotations.json and predictions.tsv for report tests."""
    ann = tmp_path / "annotations.json"
    ann.write_text(
        json.dumps(
            [
                {
                    "contig_id": "p1",
                    "mobility": {
                        "mobility_class": "conjugative",
                        "replicon_type": "IncP-1alpha",
                        "relaxase_type": "MOBP",
                        "mpf_type": "MPF_T",
                    },
                    "arg_hits": [
                        {
                            "gene_name": "NDM-6",
                            "aro_accession": "ARO:3002356",
                            "amr_family": "NDM beta-lactamase",
                            "drug_class": "carbapenem antibiotic",
                            "resistance_mechanism": "antibiotic inactivation",
                            "identity": 99.5,
                            "coverage": 95.0,
                            "evalue": 1e-120,
                        }
                    ],
                },
                {
                    "contig_id": "p2",
                    "mobility": {
                        "mobility_class": "mobilizable",
                        "replicon_type": "ColE1",
                        "relaxase_type": "MOBQ",
                        "mpf_type": "MPF_F",
                    },
                    "arg_hits": [],
                },
            ]
        )
    )

    preds = tmp_path / "predictions.tsv"
    preds.write_text(
        "sequence_id\tlabel\tconfidence\tplasmid\tchromosome\tphage\tarchaea\n"
        "p1\tplasmid\t0.97\t0.97\t0.01\t0.01\t0.01\n"
        "p2\tplasmid\t0.94\t0.94\t0.02\t0.02\t0.02\n"
        "c1\tchromosome\t0.91\t0.05\t0.91\t0.02\t0.02\n"
    )
    return ann, preds


def test_cli_report_produces_html(tmp_path: Path) -> None:
    """report subcommand reads existing JSON/TSV and produces valid HTML."""
    ann, preds = _write_test_files(tmp_path)
    out_html = tmp_path / "report.html"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "report",
            "--annotations",
            str(ann),
            "--predictions",
            str(preds),
            "--output",
            str(out_html),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_html.exists()
    html = out_html.read_text()
    assert "PlasFlow" in html
    assert "plotly" in html.lower()


def test_cli_report_html_contains_contig_ids(tmp_path: Path) -> None:
    """HTML report contains contig IDs from annotations."""
    ann, preds = _write_test_files(tmp_path)
    out_html = tmp_path / "report.html"

    runner = CliRunner()
    runner.invoke(
        main,
        [
            "report",
            "--annotations",
            str(ann),
            "--predictions",
            str(preds),
            "--output",
            str(out_html),
        ],
    )

    html = out_html.read_text()
    assert "p1" in html
    assert "p2" in html


def test_cli_report_html_reflects_classification_counts(tmp_path: Path) -> None:
    """HTML report reflects the classification breakdown from predictions.tsv."""
    ann, preds = _write_test_files(tmp_path)
    out_html = tmp_path / "report.html"

    runner = CliRunner()
    runner.invoke(
        main,
        [
            "report",
            "--annotations",
            str(ann),
            "--predictions",
            str(preds),
            "--output",
            str(out_html),
        ],
    )

    html = out_html.read_text()
    # predictions.tsv has 3 sequences: 2 plasmids + 1 chromosome
    assert "plasmid" in html
    assert "chromosome" in html


def test_cli_report_empty_annotations(tmp_path: Path) -> None:
    """report with zero plasmid annotations (chromosome-only run) exits cleanly."""
    ann = tmp_path / "annotations.json"
    ann.write_text("[]")
    preds = tmp_path / "predictions.tsv"
    preds.write_text(
        "sequence_id\tlabel\tconfidence\tplasmid\tchromosome\tphage\tarchaea\n"
        "c1\tchromosome\t0.95\t0.02\t0.95\t0.02\t0.01\n"
    )
    out_html = tmp_path / "report.html"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "report",
            "--annotations",
            str(ann),
            "--predictions",
            str(preds),
            "--output",
            str(out_html),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_html.exists()
    assert "PlasFlow" in out_html.read_text()


# ---------------------------------------------------------------------------
# 6. Error handling
# ---------------------------------------------------------------------------


def test_cli_run_missing_model_exits_nonzero(tmp_path: Path) -> None:
    """CLI exits non-zero when model file does not exist."""
    fasta = tmp_path / "input.fasta"
    _write_fasta(fasta, {"p1": _SEQ})

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--input",
            str(fasta),
            "--output",
            str(tmp_path / "out"),
            "--model",
            str(tmp_path / "nonexistent.pt"),
        ],
    )
    assert result.exit_code != 0


def test_cli_run_missing_card_db_exits_nonzero(tmp_path: Path) -> None:
    """CLI exits non-zero when CARD database file does not exist."""
    fasta = tmp_path / "input.fasta"
    _write_fasta(fasta, {"p1": _SEQ})
    model = tmp_path / "mlp.pt"
    model.write_text("mock")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--input",
            str(fasta),
            "--output",
            str(tmp_path / "out"),
            "--model",
            str(model),
            "--card-db",
            str(tmp_path / "missing.dmnd"),
        ],
    )
    assert result.exit_code != 0


def test_cli_version_output(tmp_path: Path) -> None:
    """plasflow2 --version outputs the package version."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "plasflow2" in result.output.lower()


def test_cli_help_available_for_all_subcommands() -> None:
    """All subcommands respond to --help without error."""
    runner = CliRunner()
    for subcommand in ("run", "classify", "annotate", "report"):
        result = runner.invoke(main, [subcommand, "--help"])
        assert result.exit_code == 0, f"{subcommand} --help failed: {result.output}"
        assert subcommand in result.output or "Usage" in result.output
