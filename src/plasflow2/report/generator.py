"""HTML report generator.

Week 4 — Days 20 + 26 + 27 implementation.

Produces a single self-contained HTML file with:
  - Summary stats panel
  - Classification pie chart (Plotly)
  - ARG bar chart per drug class (Plotly)
  - AMR risk score histogram (Plotly)
  - Contig length vs risk score scatter plot (Plotly)
  - Drug-class co-occurrence heatmap (Plotly)
  - Per-plasmid detail table with sortable columns (DataTables.js via CDN)
  - Risk-tier filter buttons (high / medium / low / all)
  - Taxonomy column (LCA result from DIAMOND + GTDB)

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
    body { font-family: -apple-system, Arial, sans-serif; margin: 24px; color: #333; background: #fafafa; }
    h1 { color: #2c6fad; margin-bottom: 4px; }
    h2 { color: #444; margin-top: 36px; border-bottom: 2px solid #e0e8f5; padding-bottom: 6px; }
    .meta { color: #666; font-size: 0.92rem; margin-bottom: 20px; }
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin: 20px 0; }
    .stat-card { background: #fff; border-left: 4px solid #2c6fad; padding: 14px 16px; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
    .stat-card h3 { margin: 0 0 6px; font-size: 0.78rem; text-transform: uppercase; color: #777; letter-spacing: .5px; }
    .stat-card p  { margin: 0; font-size: 1.7rem; font-weight: 700; color: #2c6fad; }
    .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 24px 0; }
    .charts-row-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin: 24px 0; }
    .chart-box { background: #fff; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.08); padding: 4px; min-height: 320px; }
    table.dataTable { width: 100% !important; }
    table.dataTable tbody tr:hover { background-color: #f0f6ff; }
    .risk-high   { color: #c0392b; font-weight: bold; }
    .risk-medium { color: #e67e22; font-weight: bold; }
    .risk-low    { color: #27ae60; font-weight: bold; }
    .filter-bar  { margin: 12px 0 8px; display: flex; gap: 8px; align-items: center; }
    .filter-btn  { padding: 6px 16px; border: none; border-radius: 20px; cursor: pointer;
                   font-size: 0.85rem; font-weight: 600; transition: opacity .15s; }
    .filter-btn:hover { opacity: 0.85; }
    .filter-btn.active { outline: 3px solid #333; }
    .btn-all    { background: #e0e0e0; color: #333; }
    .btn-high   { background: #c0392b; color: #fff; }
    .btn-medium { background: #e67e22; color: #fff; }
    .btn-low    { background: #27ae60; color: #fff; }
    .tax-label  { font-size: 0.82rem; color: #555; max-width: 200px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .no-data-note { color: #888; font-style: italic; font-size: 0.9rem; margin: 8px 0; }
    footer { margin-top: 40px; color: #aaa; font-size: 0.8rem; border-top: 1px solid #e5e5e5; padding-top: 12px; }
  </style>
</head>
<body>
  <h1>PlasFlow v2 — Analysis Report</h1>
  <p class="meta">
    Input: <code>{{ input_file }}</code> &nbsp;|&nbsp;
    Sequences: <strong>{{ total }}</strong> &nbsp;|&nbsp;
    Plasmids: <strong>{{ num_plasmids }}</strong> &nbsp;|&nbsp;
    ARGs: <strong>{{ total_args }}</strong>
    {% if tax_classified is defined and tax_classified > 0 %}
    &nbsp;|&nbsp; Taxonomy-classified: <strong>{{ tax_classified }}</strong>
    {% endif %}
  </p>

  <div class="stats-grid">
    {% for label, count in class_counts.items() %}
    <div class="stat-card"><h3>{{ label }}</h3><p>{{ count }}</p></div>
    {% endfor %}
  </div>

  <h2>Classification & ARG Overview</h2>
  <div class="charts-row-3">
    <div id="pie-chart"    class="chart-box"></div>
    <div id="arg-chart"    class="chart-box"></div>
    <div id="risk-chart"   class="chart-box"></div>
  </div>

  {% if has_scatter %}
  <h2>Contig Length vs Risk Score</h2>
  <div class="charts-row">
    <div id="scatter-chart" class="chart-box" style="min-height:350px;"></div>
    <div id="tax-chart"     class="chart-box" style="min-height:350px;"></div>
  </div>
  {% endif %}

  {% if has_cooccurrence %}
  <h2>Drug-Class Co-occurrence</h2>
  <p style="color:#666;font-size:.88rem;margin:-8px 0 12px;">
    Each cell shows how many plasmid contigs carry both drug classes simultaneously.
    Darker = more co-occurrence. Only contigs with ≥2 drug classes are included.
  </p>
  <div id="cooccurrence-chart" class="chart-box" style="min-height:420px;"></div>
  {% endif %}

  <h2>Plasmid Detail</h2>
  <div class="filter-bar">
    <span style="font-size:.85rem;color:#555;">Filter by risk tier:</span>
    <button class="filter-btn btn-all active"    id="btn-all"    onclick="filterRisk('all')">All</button>
    <button class="filter-btn btn-high"   id="btn-high"   onclick="filterRisk('high')">High (&ge;7)</button>
    <button class="filter-btn btn-medium" id="btn-medium" onclick="filterRisk('medium')">Medium (4–6)</button>
    <button class="filter-btn btn-low"    id="btn-low"    onclick="filterRisk('low')">Low (0–3)</button>
  </div>
  {% if plasmid_rows %}
  <table id="plasmid-table" class="display">
    <thead>
      <tr>
        <th>Contig</th>
        <th>Length (bp)</th>
        <th>Confidence</th>
        <th>ARGs</th>
        <th>Drug Classes</th>
        <th>Mobility</th>
        <th>Replicon</th>
        <th>Risk Score</th>
        <th>Taxonomy (LCA)</th>
        <th>Risk Evidence</th>
      </tr>
    </thead>
    <tbody>
      {% for row in plasmid_rows %}
      <tr data-risk-tier="{% if row.risk_score >= 7 %}high{% elif row.risk_score >= 4 %}medium{% else %}low{% endif %}">
        <td>{{ row.contig_id }}</td>
        <td>{{ row.contig_length }}</td>
        <td>{{ "%.3f" | format(row.confidence) }}</td>
        <td>{{ row.num_args }}</td>
        <td>{{ row.drug_classes }}</td>
        <td>{{ row.mobility_class }}</td>
        <td>{{ row.replicon_type }}</td>
        <td class="{% if row.risk_score >= 7 %}risk-high{% elif row.risk_score >= 4 %}risk-medium{% else %}risk-low{% endif %}">
          {{ row.risk_score }}
        </td>
        <td class="tax-label" title="{{ row.taxonomy }}">{{ row.taxonomy }}</td>
        <td>{{ row.risk_evidence }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% else %}
  <p class="no-data-note">No plasmid contigs detected in this run.</p>
  {% endif %}

  <footer>Generated by PlasFlow v2 &mdash; open in any modern browser, no server required.</footer>

  <script>
    // ---- Plotly charts ----
    var pieData = {{ pie_data | tojson }};
    Plotly.newPlot('pie-chart', pieData.data, pieData.layout, {responsive: true});

    var argData = {{ arg_data | tojson }};
    Plotly.newPlot('arg-chart', argData.data, argData.layout, {responsive: true});

    var riskData = {{ risk_data | tojson }};
    Plotly.newPlot('risk-chart', riskData.data, riskData.layout, {responsive: true});

    {% if has_scatter %}
    var scatterData = {{ scatter_data | tojson }};
    Plotly.newPlot('scatter-chart', scatterData.data, scatterData.layout, {responsive: true});

    var taxData = {{ tax_bar_data | tojson }};
    Plotly.newPlot('tax-chart', taxData.data, taxData.layout, {responsive: true});
    {% endif %}

    {% if has_cooccurrence %}
    var coData = {{ cooccurrence_data | tojson }};
    Plotly.newPlot('cooccurrence-chart', coData.data, coData.layout, {responsive: true});
    {% endif %}

    // ---- DataTable ----
    var table = null;
    $(document).ready(function() {
      table = $('#plasmid-table').DataTable({order: [[7, 'desc']], pageLength: 25});
    });

    // ---- Risk tier filter ----
    function filterRisk(tier) {
      // Update button styles
      ['all','high','medium','low'].forEach(function(t) {
        document.getElementById('btn-' + t).classList.toggle('active', t === tier);
      });
      if (!table) return;
      // Use DataTables search with a custom filter plugin
      $.fn.dataTable.ext.search = $.fn.dataTable.ext.search.filter(function(f) {
        return f.__riskFilter !== true;
      });
      if (tier !== 'all') {
        var fn = function(settings, data, dataIndex) {
          var row = table.row(dataIndex).node();
          return $(row).data('risk-tier') === tier;
        };
        fn.__riskFilter = true;
        $.fn.dataTable.ext.search.push(fn);
      }
      table.draw();
    }
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
    contig_length: int
    confidence: float
    num_args: int
    drug_classes: str  # semicolon-separated unique drug classes
    mobility_class: str
    replicon_type: str
    risk_score: int
    taxonomy: str  # LCA display string, e.g. "genus: g__Klebsiella"
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
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
        },
    }


def _build_arg_chart(arg_hits: list) -> dict:
    """Plotly horizontal bar chart: ARG count per drug class."""
    drug_class_counts: Counter[str] = Counter()
    for hit in arg_hits:
        for dc in hit.drug_class.split(";"):
            dc = dc.strip()
            if dc and dc != "unknown":
                drug_class_counts[dc] += 1

    if not drug_class_counts:
        return {
            "data": [{"type": "bar", "x": [], "y": [], "orientation": "h"}],
            "layout": {
                "title": {"text": "ARG Drug Classes (none detected)", "font": {"size": 14}},
                "margin": {"t": 50, "b": 40, "l": 180, "r": 20},
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
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
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
        },
    }


def _build_risk_histogram(risk_scores: list[int]) -> dict:
    """Plotly histogram of risk scores (0–10), colour-coded by tier."""
    bar_colors = []
    counts_by_score = Counter(risk_scores)
    score_range = list(range(11))
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
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
        },
    }


def _build_scatter_data(plasmid_rows: list[PlasmidRow]) -> dict:
    """Plotly scatter: contig length vs risk score, coloured by mobility class."""
    mobility_classes = sorted({r.mobility_class for r in plasmid_rows})
    palette = ["#2c6fad", "#c0392b", "#27ae60", "#e67e22", "#8e44ad", "#16a085", "#d35400"]
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(mobility_classes)}

    traces = []
    for mob in mobility_classes:
        rows = [r for r in plasmid_rows if r.mobility_class == mob]
        if not rows:
            continue
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": mob,
                "x": [r.contig_length for r in rows],
                "y": [r.risk_score for r in rows],
                "text": [
                    f"{r.contig_id}<br>Risk: {r.risk_score}<br>ARGs: {r.num_args}<br>{r.taxonomy}"
                    for r in rows
                ],
                "hovertemplate": "%{text}<extra></extra>",
                "marker": {
                    "color": color_map[mob],
                    "size": 7,
                    "opacity": 0.75,
                    "line": {"width": 0.5, "color": "#fff"},
                },
            }
        )

    return {
        "data": traces,
        "layout": {
            "title": {"text": "Contig Length vs Risk Score", "font": {"size": 14}},
            "xaxis": {"title": "Contig length (bp)", "type": "log"},
            "yaxis": {"title": "Risk Score (0–10)", "dtick": 1, "range": [-0.5, 10.5]},
            "legend": {"title": {"text": "Mobility"}},
            "margin": {"t": 50, "b": 60, "l": 60, "r": 20},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
        },
    }


def _build_taxonomy_bar(plasmid_rows: list[PlasmidRow]) -> dict:
    """Plotly bar chart: top-15 taxonomy assignments for plasmid contigs."""
    tax_counts: Counter[str] = Counter()
    for r in plasmid_rows:
        label = r.taxonomy if r.taxonomy and r.taxonomy != "—" else "unclassified"
        tax_counts[label] += 1

    top15 = tax_counts.most_common(15)
    if not top15:
        return {
            "data": [{"type": "bar", "x": [], "y": []}],
            "layout": {
                "title": {"text": "Top Taxonomy (no data)", "font": {"size": 14}},
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
            },
        }

    labels = [item[0] for item in reversed(top15)]
    counts = [item[1] for item in reversed(top15)]

    return {
        "data": [
            {
                "type": "bar",
                "x": counts,
                "y": labels,
                "orientation": "h",
                "marker": {"color": "#8e44ad"},
            }
        ],
        "layout": {
            "title": {"text": "Top Taxonomy (plasmid contigs)", "font": {"size": 14}},
            "xaxis": {"title": "Contig count"},
            "margin": {"t": 50, "b": 40, "l": 220, "r": 20},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
        },
    }


def _build_drug_cooccurrence_heatmap(plasmid_results: list) -> dict:
    """Plotly heatmap: drug-class co-occurrence across plasmid contigs.

    For every pair of drug classes (A, B), count how many distinct plasmid
    contigs carry at least one ARG from each class simultaneously.  The matrix
    is symmetric; diagonal cells show the total number of contigs carrying that
    class at all.

    Args:
        plasmid_results: List of ContigResult objects from the pipeline.

    Returns:
        Plotly figure dict, or an empty placeholder if there are fewer than
        2 drug classes present.
    """
    # Build per-contig drug-class sets
    contig_classes: list[frozenset[str]] = []
    for cr in plasmid_results:
        classes: set[str] = set()
        for hit in cr.arg_hits:
            for dc in hit.drug_class.split(";"):
                dc = dc.strip()
                if dc and dc not in ("unknown", ""):
                    classes.add(dc)
        if classes:
            contig_classes.append(frozenset(classes))

    # Collect all unique drug classes (sorted for stable axis order)
    all_classes = sorted({dc for classes in contig_classes for dc in classes})

    if len(all_classes) < 2:
        return {
            "data": [],
            "layout": {
                "title": {
                    "text": "Drug-Class Co-occurrence (insufficient data)",
                    "font": {"size": 14},
                },
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
            },
        }

    n = len(all_classes)
    # Build n×n count matrix
    matrix = [[0] * n for _ in range(n)]
    for fset in contig_classes:
        for i, ci in enumerate(all_classes):
            if ci not in fset:
                continue
            for j, cj in enumerate(all_classes):
                if cj in fset:
                    matrix[i][j] += 1

    # Shorten long drug-class names for axis labels
    def _short(label: str, maxlen: int = 28) -> str:
        return label if len(label) <= maxlen else label[: maxlen - 1] + "…"

    short_labels = [_short(c) for c in all_classes]

    # Custom hover text: "X ∩ Y: N contigs"
    hover = [
        [f"{all_classes[i]}<br>∩ {all_classes[j]}<br>{matrix[i][j]} contig(s)" for j in range(n)]
        for i in range(n)
    ]

    return {
        "data": [
            {
                "type": "heatmap",
                "z": matrix,
                "x": short_labels,
                "y": short_labels,
                "text": hover,
                "hovertemplate": "%{text}<extra></extra>",
                "colorscale": "Blues",
                "showscale": True,
                "colorbar": {"title": "Contigs", "thickness": 14},
            }
        ],
        "layout": {
            "title": {"text": "Drug-Class Co-occurrence (plasmid contigs)", "font": {"size": 14}},
            "xaxis": {
                "title": "",
                "tickangle": -40,
                "tickfont": {"size": 10},
                "automargin": True,
            },
            "yaxis": {
                "title": "",
                "tickfont": {"size": 10},
                "automargin": True,
            },
            "margin": {"t": 60, "b": 120, "l": 160, "r": 40},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
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
    all_arg_hits = [hit for cr in pipeline_result.plasmid_results for hit in cr.arg_hits]

    # Taxonomy dict (contig_id → TaxResult); may be empty if skipped
    taxonomy = getattr(pipeline_result, "taxonomy", {}) or {}

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
        # Taxonomy: prefer ContigResult.taxonomy if set, else look up in global dict
        tax = getattr(cr, "taxonomy", None) or taxonomy.get(cr.record.id)
        tax_display = tax.display if tax else "—"

        plasmid_rows.append(
            PlasmidRow(
                contig_id=cr.record.id,
                contig_length=len(cr.record.seq),
                confidence=cr.prediction.confidence,
                num_args=len(cr.arg_hits),
                drug_classes="; ".join(unique_classes) if unique_classes else "—",
                mobility_class=mob.mobility_class if mob else "unknown",
                replicon_type=mob.replicon_type if mob else "unknown",
                risk_score=cr.risk.score,
                taxonomy=tax_display,
                risk_evidence="; ".join(cr.risk.evidence) if cr.risk.evidence else "—",
            )
        )

    risk_scores = [cr.risk.score for cr in pipeline_result.plasmid_results]
    tax_classified = sum(1 for r in taxonomy.values() if r.rank != "unclassified")
    has_scatter = len(plasmid_rows) > 0

    # Drug-class co-occurrence — needs raw ContigResult objects (with .arg_hits)
    cooccurrence_data = _build_drug_cooccurrence_heatmap(pipeline_result.plasmid_results)
    # Show heatmap only when at least 2 distinct drug classes are present
    has_cooccurrence = (
        bool(cooccurrence_data.get("data"))
        and len(cooccurrence_data["data"]) > 0
        and cooccurrence_data["data"][0].get("z", [])
    )

    return {
        "input_file": input_file or str(pipeline_result.input_fasta),
        "total": pipeline_result.total_sequences,
        "num_plasmids": pipeline_result.total_plasmids,
        "total_args": pipeline_result.total_args,
        "tax_classified": tax_classified,
        "class_counts": pipeline_result.class_counts,
        "pie_data": _build_pie_data(pipeline_result.class_counts),
        "arg_data": _build_arg_chart(all_arg_hits),
        "risk_data": _build_risk_histogram(risk_scores),
        "scatter_data": _build_scatter_data(plasmid_rows) if has_scatter else {},
        "tax_bar_data": _build_taxonomy_bar(plasmid_rows) if has_scatter else {},
        "cooccurrence_data": cooccurrence_data,
        "plasmid_rows": plasmid_rows,
        "has_scatter": has_scatter,
        "has_cooccurrence": has_cooccurrence,
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
        safe_data = {
            k: v
            for k, v in report_data.items()
            if k
            not in (
                "pie_data",
                "arg_data",
                "risk_data",
                "scatter_data",
                "cooccurrence_data",
                "tax_bar_data",
                "plasmid_rows",
            )
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
