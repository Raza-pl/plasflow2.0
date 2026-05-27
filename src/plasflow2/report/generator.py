"""HTML report generator.

Week 4 — Day 20 implementation.

Produces a single self-contained HTML file with:
  - Summary stats panel
  - Classification pie chart (Plotly)
  - ARG bar chart per drug class (Plotly)
  - AMR risk score histogram (Plotly)
  - Per-plasmid detail table with sortable columns (DataTables.js via CDN)

Usage:
    from plasflow2.report.generator import build_report_data, generate_report
    data = build_report_data(pipeline_result, input_file="contigs.fasta")
    generate_report(data, output_path="report.html")
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Jinja2 HTML template (self-contained — CDN assets only)
# ---------------------------------------------------------------------------

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PlasFlow v2 Report</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
  <style>
    body { font-family: -apple-system, Arial, sans-serif; margin: 24px; color: #333; }
    h1 { color: #2c6fad; }
    h2 { color: #444; margin-top: 32px; }
    .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }
    .stat-card { background: #f5f8ff; border-left: 4px solid #2c6fad; padding: 16px; border-radius: 4px; }
    .stat-card h3 { margin: 0 0 8px; font-size: 0.85rem; text-transform: uppercase; color: #666; }
    .stat-card p  { margin: 0; font-size: 1.8rem; font-weight: 700; color: #2c6fad; }
    .charts-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin: 24px 0; }
    .chart-box { min-height: 320px; }
    table.dataTable { width: 100% !important; }
    .risk-high   { color: #c0392b; font-weight: bold; }
    .risk-medium { color: #e67e22; font-weight: bold; }
    .risk-low    { color: #27ae60; font-weight: bold; }
  </style>
</head>
<body>
  <h1>PlasFlow v2 — Analysis Report</h1>
  <p>Input: <code>{{ input_file }}</code> &nbsp;|&nbsp;
     Sequences: <strong>{{ total }}</strong> &nbsp;|&nbsp;
     Plasmids: <strong>{{ num_plasmids }}</strong> &nbsp;|&nbsp;
     ARGs: <strong>{{ total_args }}</strong></p>

  <div class="stats-grid">
    {% for label, count in class_counts.items() %}
    <div class="stat-card"><h3>{{ label }}</h3><p>{{ count }}</p></div>
    {% endfor %}
  </div>

  <div class="charts-row">
    <div id="pie-chart"  class="chart-box"></div>
    <div id="arg-chart"  class="chart-box"></div>
    <div id="risk-chart" class="chart-box"></div>
  </div>

  <h2>Plasmid Detail</h2>
  <table id="plasmid-table" class="display">
    <thead>
      <tr>
        <th>Contig</th>
        <th>Confidence</th>
        <th>ARGs</th>
        <th>Drug Classes</th>
        <th>Mobility</th>
        <th>Replicon</th>
        <th>Risk Score</th>
        <th>Risk Evidence</th>
      </tr>
    </thead>
    <tbody>
      {% for row in plasmid_rows %}
      <tr>
        <td>{{ row.contig_id }}</td>
        <td>{{ "%.3f" | format(row.confidence) }}</td>
        <td>{{ row.num_args }}</td>
        <td>{{ row.drug_classes }}</td>
        <td>{{ row.mobility_class }}</td>
        <td>{{ row.replicon_type }}</td>
        <td class="{% if row.risk_score >= 7 %}risk-high{% elif row.risk_score >= 4 %}risk-medium{% else %}risk-low{% endif %}">
          {{ row.risk_score }}
        </td>
        <td>{{ row.risk_evidence }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <script>
    var pieData = {{ pie_data | tojson }};
    Plotly.newPlot('pie-chart', pieData.data, pieData.layout, {responsive: true});

    var argData = {{ arg_data | tojson }};
    Plotly.newPlot('arg-chart', argData.data, argData.layout, {responsive: true});

    var riskData = {{ risk_data | tojson }};
    Plotly.newPlot('risk-chart', riskData.data, riskData.layout, {responsive: true});

    $(document).ready(function() {
      $('#plasmid-table').DataTable({order: [[6, 'desc']]});
    });
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PlasmidRow:
    """One row in the per-plasmid detail table."""

    contig_id: str
    confidence: float
    num_args: int
    drug_classes: str  # semicolon-separated unique drug classes
    mobility_class: str
    replicon_type: str
    risk_score: int
    risk_evidence: str  # semicolon-separated evidence strings


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _build_pie_data(class_counts: dict[str, int]) -> dict:
    """Plotly pie chart: sequence classification breakdown."""
    labels = list(class_counts.keys())
    values = list(class_counts.values())
    colors = {
        "plasmid": "#2c6fad",
        "chromosome": "#27ae60",
        "phage": "#e67e22",
        "archaea": "#8e44ad",
        "unclassified": "#95a5a6",
    }
    return {
        "data": [
            {
                "type": "pie",
                "labels": labels,
                "values": values,
                "marker": {"colors": [colors.get(lbl, "#aaa") for lbl in labels]},
                "textinfo": "label+percent",
                "hole": 0.35,
            }
        ],
        "layout": {
            "title": {"text": "Sequence Classification", "font": {"size": 14}},
            "margin": {"t": 50, "b": 20, "l": 20, "r": 20},
            "showlegend": False,
        },
    }


def _build_arg_chart(arg_hits: list) -> dict:
    """Plotly horizontal bar chart: ARG count per drug class."""
    drug_class_counts: Counter[str] = Counter()
    for hit in arg_hits:
        # drug_class may be semicolon-separated
        for dc in hit.drug_class.split(";"):
            dc = dc.strip()
            if dc and dc != "unknown":
                drug_class_counts[dc] += 1

    if not drug_class_counts:
        # Empty placeholder
        return {
            "data": [{"type": "bar", "x": [], "y": [], "orientation": "h"}],
            "layout": {
                "title": {"text": "ARG Drug Classes (none detected)", "font": {"size": 14}},
                "margin": {"t": 50, "b": 40, "l": 180, "r": 20},
            },
        }

    sorted_items = sorted(drug_class_counts.items(), key=lambda x: x[1])
    classes = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    return {
        "data": [
            {
                "type": "bar",
                "x": counts,
                "y": classes,
                "orientation": "h",
                "marker": {"color": "#c0392b"},
            }
        ],
        "layout": {
            "title": {"text": "ARGs by Drug Class", "font": {"size": 14}},
            "xaxis": {"title": "Gene count"},
            "margin": {"t": 50, "b": 40, "l": 180, "r": 20},
        },
    }


def _build_risk_histogram(risk_scores: list[int]) -> dict:
    """Plotly histogram of risk scores (0–10)."""
    # Colour each bar: >=7 red, 4-6 orange, 0-3 green
    bar_colors = []
    counts_by_score = Counter(risk_scores)
    score_range = list(range(11))  # 0..10
    y_vals = [counts_by_score.get(s, 0) for s in score_range]
    for s in score_range:
        if s >= 7:
            bar_colors.append("#c0392b")
        elif s >= 4:
            bar_colors.append("#e67e22")
        else:
            bar_colors.append("#27ae60")

    return {
        "data": [
            {
                "type": "bar",
                "x": score_range,
                "y": y_vals,
                "marker": {"color": bar_colors},
            }
        ],
        "layout": {
            "title": {"text": "Risk Score Distribution", "font": {"size": 14}},
            "xaxis": {"title": "Risk Score (0–10)", "dtick": 1},
            "yaxis": {"title": "Plasmid count"},
            "margin": {"t": 50, "b": 50, "l": 50, "r": 20},
        },
    }


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_report_data(pipeline_result, input_file: str = "") -> dict:
    """Convert a PipelineResult into the template data dict.

    Args:
        pipeline_result: :class:`~plasflow2.pipeline.PipelineResult`.
        input_file: Display name for the input FASTA (shown in report header).

    Returns:
        Dict suitable for passing to :func:`generate_report`.
    """
    # Collect all ARG hits across plasmid contigs
    all_arg_hits = [hit for cr in pipeline_result.plasmid_results for hit in cr.arg_hits]

    # Build per-plasmid table rows
    plasmid_rows: list[PlasmidRow] = []
    for cr in pipeline_result.plasmid_results:
        unique_classes = sorted(
            {
                dc.strip()
                for hit in cr.arg_hits
                for dc in hit.drug_class.split(";")
                if dc.strip() and dc.strip() != "unknown"
            }
        )
        mob = cr.mobility
        plasmid_rows.append(
            PlasmidRow(
                contig_id=cr.record.id,
                confidence=cr.prediction.confidence,
                num_args=len(cr.arg_hits),
                drug_classes="; ".join(unique_classes) if unique_classes else "—",
                mobility_class=mob.mobility_class if mob else "unknown",
                replicon_type=mob.replicon_type if mob else "unknown",
                risk_score=cr.risk.score,
                risk_evidence="; ".join(cr.risk.evidence) if cr.risk.evidence else "—",
            )
        )

    risk_scores = [cr.risk.score for cr in pipeline_result.plasmid_results]

    return {
        "input_file": input_file or str(pipeline_result.input_fasta),
        "total": pipeline_result.total_sequences,
        "num_plasmids": pipeline_result.total_plasmids,
        "total_args": pipeline_result.total_args,
        "class_counts": pipeline_result.class_counts,
        "pie_data": _build_pie_data(pipeline_result.class_counts),
        "arg_data": _build_arg_chart(all_arg_hits),
        "risk_data": _build_risk_histogram(risk_scores),
        "plasmid_rows": plasmid_rows,
    }


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def generate_report(
    report_data: dict,
    output_path: Path | str,
) -> Path:
    """Render the HTML report from structured data.

    Args:
        report_data: Dict produced by :func:`build_report_data`.
        output_path: Destination .html file.

    Returns:
        Path to the written HTML file.
    """
    try:
        from jinja2 import Environment  # type: ignore[import]

        env = Environment(autoescape=False)
        env.filters["tojson"] = json.dumps
        tmpl = env.from_string(_TEMPLATE)
        html = tmpl.render(**report_data)
    except ImportError:
        logger.warning("jinja2 not installed — writing JSON placeholder report")
        # Produce a minimal but valid HTML fallback that doesn't crash
        safe_data = {
            k: v
            for k, v in report_data.items()
            if k not in ("pie_data", "arg_data", "risk_data", "plasmid_rows")
        }
        html = (
            "<html><body><h1>PlasFlow v2 Report</h1>"
            f"<pre>{json.dumps(safe_data, indent=2)}</pre>"
            "</body></html>"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", output_path)
    return output_path
