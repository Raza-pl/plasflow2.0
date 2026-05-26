"""HTML report generator.

Week 4 — Days 23–24 implementation target.

Produces a single self-contained HTML file with:
  - Summary stats panel
  - Classification pie chart (Plotly)
  - ARG bar chart per drug class (Plotly)
  - AMR risk histogram (Plotly)
  - Per-plasmid detail table with sortable columns (DataTables.js via CDN)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimal Jinja2 template (inline for now — move to report/templates/ in Week 4)
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
    .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }
    .stat-card { background: #f5f8ff; border-left: 4px solid #2c6fad; padding: 16px; border-radius: 4px; }
    .stat-card h3 { margin: 0 0 8px; font-size: 0.85rem; text-transform: uppercase; color: #666; }
    .stat-card p  { margin: 0; font-size: 1.8rem; font-weight: 700; color: #2c6fad; }
    .charts-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin: 24px 0; }
    table.dataTable { width: 100% !important; }
  </style>
</head>
<body>
  <h1>PlasFlow v2 — Analysis Report</h1>
  <p>Input: <code>{{ input_file }}</code> &nbsp;|&nbsp; Sequences: <strong>{{ total }}</strong></p>

  <div class="stats-grid">
    {% for label, count in class_counts.items() %}
    <div class="stat-card"><h3>{{ label }}</h3><p>{{ count }}</p></div>
    {% endfor %}
    <div class="stat-card"><h3>Unclassified</h3><p>{{ unclassified }}</p></div>
  </div>

  <div class="charts-row">
    <div id="pie-chart"></div>
    <div id="arg-chart"></div>
    <div id="risk-chart"></div>
  </div>

  <h2>Plasmid Detail</h2>
  <table id="plasmid-table" class="display">
    <thead>
      <tr>
        <th>Contig</th><th>Confidence</th><th>ARGs</th>
        <th>Mobility</th><th>Replicon</th><th>Risk Score</th>
      </tr>
    </thead>
    <tbody>
      {% for row in plasmid_rows %}
      <tr>
        <td>{{ row.contig_id }}</td>
        <td>{{ "%.3f"|format(row.confidence) }}</td>
        <td>{{ row.num_args }}</td>
        <td>{{ row.mobility_class }}</td>
        <td>{{ row.replicon_type }}</td>
        <td><strong>{{ row.risk_score }}</strong></td>
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

    $(document).ready(function() { $('#plasmid-table').DataTable(); });
  </script>
</body>
</html>
"""


def generate_report(
    report_data: dict,
    output_path: Path | str,
) -> Path:
    """Render the HTML report from structured data.

    Args:
        report_data: Dict produced by the pipeline with keys:
            input_file, total, class_counts, unclassified,
            pie_data, arg_data, risk_data, plasmid_rows.
        output_path: Destination .html file.

    Returns:
        Path to the written HTML file.

    TODO (Days 23–24):
        - Move template to report/templates/report.html.j2
        - Add per-plasmid detail modal / drilldown pages.
        - Build pie_data / arg_data / risk_data Plotly dicts from raw results.
    """
    try:
        from jinja2 import Environment  # type: ignore[import]
        env = Environment()
        env.filters["tojson"] = json.dumps
        tmpl = env.from_string(_TEMPLATE)
        html = tmpl.render(**report_data)
    except ImportError:
        logger.warning("jinja2 not installed — writing placeholder report")
        html = f"<html><body><pre>{json.dumps(report_data, indent=2)}</pre></body></html>"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", output_path)
    return output_path
