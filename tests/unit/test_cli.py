"""Unit tests for the PlasFlow v2 CLI (cli.py).

Uses Click's CliRunner so no actual DIAMOND, mob_typer, or model weights needed.
All external I/O is mocked at the pipeline / module level.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
from plasflow2.annotate.args import ARGHit
from plasflow2.annotate.mobility import MobilityResult
from plasflow2.classify.predict import Prediction
from plasflow2.cli import main
from plasflow2.pipeline import ContigResult, PipelineResult
from plasflow2.risk.scorer import RiskScore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEQ = "ACGT" * 500


def _prediction(seq_id: str, label: str) -> Prediction:
    scores = {k: 0.0 for k in ("plasmid", "chromosome", "phage", "archaea")}
    scores[label] = 0.95
    return Prediction(sequence_id=seq_id, label=label, confidence=0.95, scores=scores)


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


def _mob(contig_id: str) -> MobilityResult:
    return MobilityResult(
        contig_id=contig_id,
        mobility_class="conjugative",
        replicon_type="IncP-1alpha",
        relaxase_type="MOBP",
        mpf_type="MPF_T",
    )


def _risk(contig_id: str) -> RiskScore:
    return RiskScore(
        contig_id=contig_id,
        score=7,
        evidence=["Conjugative (+3)", "IncP-1alpha (+2)"],
        mobility_score=3,
        arg_score=2,
        replicon_score=2,
        context_score=0,
    )


def _make_pipeline_result(tmp_path: Path) -> PipelineResult:
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    record = SeqRecord(Seq(_SEQ), id="p1", description="")
    cr = ContigResult(
        record=record,
        prediction=_prediction("p1", "plasmid"),
        arg_hits=[_arg_hit("p1")],
        mobility=_mob("p1"),
        risk=_risk("p1"),
    )
    return PipelineResult(
        input_fasta=tmp_path / "contigs.fasta",
        all_predictions=[_prediction("p1", "plasmid"), _prediction("c1", "chromosome")],
        plasmid_results=[cr],
    )


# ---------------------------------------------------------------------------
# plasflow2 --version
# ---------------------------------------------------------------------------


def test_version() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "plasflow2" in result.output.lower()


# ---------------------------------------------------------------------------
# plasflow2 run
# ---------------------------------------------------------------------------


def test_run_produces_outputs(tmp_path: Path) -> None:
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(f">p1\n{_SEQ}\n>c1\n{_SEQ}\n")
    model = tmp_path / "mlp.pt"
    model.write_text("mock")
    card_db = tmp_path / "card.dmnd"
    card_db.write_text("mock")
    aro_index = tmp_path / "aro_index.tsv"
    aro_index.write_text("mock")
    out = tmp_path / "out"

    pipeline_result = _make_pipeline_result(tmp_path)

    runner = CliRunner()
    with patch("plasflow2.cli.run_pipeline", return_value=pipeline_result):
        with patch("plasflow2.cli.load_fasta") as mock_load:
            from Bio.Seq import Seq
            from Bio.SeqRecord import SeqRecord

            mock_load.return_value = [
                SeqRecord(Seq(_SEQ), id="p1", description=""),
                SeqRecord(Seq(_SEQ), id="c1", description=""),
            ]
            with patch("plasflow2.cli.write_fasta"):
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
    assert (out / "predictions.tsv").exists()
    assert (out / "annotations.json").exists()
    assert (out / "report.html").exists()


def test_run_missing_model_raises(tmp_path: Path) -> None:
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(f">p1\n{_SEQ}\n")
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


def test_run_missing_card_db_raises(tmp_path: Path) -> None:
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(f">p1\n{_SEQ}\n")
    model = tmp_path / "model.pt"
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


# ---------------------------------------------------------------------------
# plasflow2 classify
# ---------------------------------------------------------------------------


def test_classify_writes_tsv(tmp_path: Path) -> None:
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(f">p1\n{_SEQ}\n>c1\n{_SEQ}\n")
    model = tmp_path / "model.pt"
    model.write_text("mock")
    out_tsv = tmp_path / "preds.tsv"

    preds = [_prediction("p1", "plasmid"), _prediction("c1", "chromosome")]

    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    runner = CliRunner()
    with patch(
        "plasflow2.cli.load_fasta",
        return_value=[
            SeqRecord(Seq(_SEQ), id="p1", description=""),
            SeqRecord(Seq(_SEQ), id="c1", description=""),
        ],
    ):
        with patch("plasflow2.cli.predict", return_value=preds):
            result = runner.invoke(
                main,
                [
                    "classify",
                    "--input",
                    str(fasta),
                    "--output",
                    str(out_tsv),
                    "--model",
                    str(model),
                ],
            )

    assert result.exit_code == 0, result.output
    assert out_tsv.exists()
    rows = list(csv.DictReader(out_tsv.open(), delimiter="\t"))
    assert len(rows) == 2
    assert rows[0]["contig_id"] == "p1"
    assert rows[0]["label"] == "plasmid"


def test_classify_tsv_has_correct_columns(tmp_path: Path) -> None:
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(f">s1\n{_SEQ}\n")
    model = tmp_path / "model.pt"
    model.write_text("mock")
    out_tsv = tmp_path / "preds.tsv"

    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    runner = CliRunner()
    with patch(
        "plasflow2.cli.load_fasta", return_value=[SeqRecord(Seq(_SEQ), id="s1", description="")]
    ):
        with patch("plasflow2.cli.predict", return_value=[_prediction("s1", "plasmid")]):
            runner.invoke(
                main,
                [
                    "classify",
                    "--input",
                    str(fasta),
                    "--output",
                    str(out_tsv),
                    "--model",
                    str(model),
                ],
            )

    with open(out_tsv) as fh:
        header = fh.readline().strip().split("\t")
    assert header == [
        "contig_id",
        "label",
        "confidence",
        "plasmid_score",
        "chromosome_score",
        "phage_score",
        "archaea_score",
    ]


# ---------------------------------------------------------------------------
# plasflow2 annotate
# ---------------------------------------------------------------------------


def test_annotate_writes_json(tmp_path: Path) -> None:
    fasta = tmp_path / "plasmids.fasta"
    fasta.write_text(f">p1\n{_SEQ}\n")
    card_db = tmp_path / "card.dmnd"
    card_db.write_text("mock")
    aro_index = tmp_path / "aro_index.tsv"
    aro_index.write_text("mock")
    out_dir = tmp_path / "ann_out"

    runner = CliRunner()
    with patch("plasflow2.cli.annotate_contigs", return_value=[_arg_hit("p1")]):
        with patch("plasflow2.cli.annotate_mobility", return_value=[_mob("p1")]):
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


def test_annotate_skip_mobility(tmp_path: Path) -> None:
    fasta = tmp_path / "plasmids.fasta"
    fasta.write_text(f">p1\n{_SEQ}\n")
    card_db = tmp_path / "card.dmnd"
    card_db.write_text("mock")
    aro_index = tmp_path / "aro_index.tsv"
    aro_index.write_text("mock")

    runner = CliRunner()
    with patch("plasflow2.cli.annotate_contigs", return_value=[]):
        with patch("plasflow2.cli.annotate_mobility") as mock_mob:
            runner.invoke(
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
            mock_mob.assert_not_called()


# ---------------------------------------------------------------------------
# plasflow2 report
# ---------------------------------------------------------------------------


def test_report_cmd_produces_html(tmp_path: Path) -> None:
    # Write minimal annotations.json
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
                }
            ]
        )
    )

    # Write minimal predictions.tsv (new 27-column format)
    preds = tmp_path / "predictions.tsv"
    preds.write_text(
        "contig_id\tlength\tlabel\tconfidence\tplasmid_score\tchromosome_score\tphage_score\tarchaea_score\t"
        "taxonomy\ttaxonomy_rank\ttaxonomy_lineage\t"
        "num_args\tdrug_classes\targ_sources\t"
        "mobility_class\treplicon_type\trelaxase_type\tmpf_type\t"
        "risk_score\tmobility_score\targ_score\treplicon_score\t"
        "context_score\thost_score\trisk_evidence\teskape_host\teskape_genus\n"
        "p1\t5000\tplasmid\t0.95\t0.95\t0.02\t0.02\t0.01\t\t\t\t0\t\t\t\t\t\t\t0\t0\t0\t0\t0\t0\t\tFalse\t\n"
        "c1\t3000\tchromosome\t0.90\t0.05\t0.90\t0.03\t0.02\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\n"
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
