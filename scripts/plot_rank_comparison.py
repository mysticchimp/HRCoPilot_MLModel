"""Render an interactive bump chart comparing the three candidate rankings for the
HR-Assistant JD: original single-LLM (`fit_0_10`), the ML pipeline, and the blind
two-judge consensus.

Positions are computed WITHIN the blind-judged pool (the top-50 union, ~78 candidates)
so all three rankings share one 1..N axis; the raw ranks are shown on hover. Output is a
self-contained HTML file (Plotly via CDN).

    COPILOT_SKIP_CLI_DOWNLOAD=1 uv run python scripts/plot_rank_comparison.py
    # then open evals/reports/rank_comparison.html in a browser
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

JUDGMENTS = "evals/judgments/blind_judgments_hr_assistant.csv"
OUT = "evals/reports/rank_comparison.html"


def main():
    j = pd.read_csv(JUDGMENTS)
    # dense position within the judged pool so all three rankings live on one 1..N axis
    j["pipeline_pos"] = j["pipeline_rank"].rank(method="min").astype(int)
    j["llm_pos"] = j["llm_rank"].rank(method="min").astype(int)
    j["judge_pos"] = j["consensus_rank"].rank(method="min").astype(int)
    n = len(j)

    corr = {
        "ML Pipeline \u2194 Blind Judges": round(float(j["pipeline_rank"].corr(j["consensus_rank"], method="spearman")), 3),
        "Original LLM \u2194 Blind Judges": round(float(j["llm_rank"].corr(j["consensus_rank"], method="spearman")), 3),
        "ML Pipeline \u2194 Original LLM": round(float(j["pipeline_rank"].corr(j["llm_rank"], method="spearman")), 3),
    }

    records = [{
        "id": r["candidate_id"],
        "title": str(r.get("current_title", "")),
        "pipeline_rank": int(r["pipeline_rank"]),
        "llm_rank": int(r["llm_rank"]),
        "judge_rank": int(r["consensus_rank"]),
        "pipeline_pos": int(r["pipeline_pos"]),
        "llm_pos": int(r["llm_pos"]),
        "judge_pos": int(r["judge_pos"]),
        "judge_score": float(r["judge_mean_score"]),
    } for _, r in j.iterrows()]

    html = (_HTML.replace("__DATA__", json.dumps(records))
                 .replace("__CORR__", json.dumps(corr))
                 .replace("__N__", str(n)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(html)
    print(f"wrote {OUT}  ({n} candidates)")
    for k, v in corr.items():
        print(f"  Spearman {k}: {v}")


_HTML = r'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Rank comparison — HR Assistant</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #1a1a1a; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  p  { color: #555; margin-top: 0; max-width: 900px; }
  .corr { display: flex; gap: 12px; margin: 14px 0; flex-wrap: wrap; }
  .corr div { background: #f4f4f6; border-radius: 8px; padding: 8px 12px; font-size: 13px; }
  .corr b { font-size: 15px; }
  #chart { width: 100%; height: 820px; }
</style>
</head>
<body>
<h1>Candidate ranking comparison — HR Assistant (Prime Focus)</h1>
<p>Each line is one candidate across the three rankings; <b>position 1 (top) = best</b>.
Positions are within the __N__-candidate blind-judged pool so all three share one scale
(hover shows the raw ranks). A near-flat line = the three rankings agree on that candidate;
a steep slope = they disagree. Lines are coloured by blind-judge position
(<span style="color:#472d7b">dark = judges' favourites</span> →
<span style="color:#c8b900">light = judges' lowest</span>).</p>
<div class="corr" id="corr"></div>
<div id="chart"></div>
<script>
const DATA = __DATA__;
const CORR = __CORR__;
const N = __N__;
const COLS = ["Original LLM", "ML Pipeline", "Blind Judges"];

document.getElementById("corr").innerHTML =
  '<div style="background:#eef;">Spearman rank correlation (over the judged pool):</div>' +
  Object.entries(CORR).map(([k, v]) => `<div>${k}<br><b>&rho; = ${v}</b></div>`).join("");

// viridis-style colour by judge position (1 = best -> dark purple)
function color(pos) {
  const t = (pos - 1) / (N - 1);
  const stops = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  const s = t * (stops.length - 1);
  const i = Math.min(Math.floor(s), stops.length - 2);
  const f = s - i;
  const c = stops[i].map((a, k) => Math.round(a + f * (stops[i + 1][k] - a)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

const traces = DATA.map(d => ({
  x: COLS,
  y: [d.llm_pos, d.pipeline_pos, d.judge_pos],
  mode: "lines+markers",
  line: { color: color(d.judge_pos), width: 1.3 },
  marker: { size: 5, color: color(d.judge_pos) },
  opacity: 0.6,
  hoverinfo: "text",
  text: COLS.map(() =>
    `${d.id}<br>${d.title}<br>` +
    `pipeline #${d.pipeline_rank} &middot; LLM #${d.llm_rank} &middot; judge #${d.judge_rank}` +
    `<br>judge score ${d.judge_score.toFixed(1)}`),
  showlegend: false,
}));

Plotly.newPlot("chart", traces, {
  margin: { l: 70, r: 30, t: 20, b: 40 },
  xaxis: { side: "top", tickfont: { size: 15 }, fixedrange: true },
  yaxis: { title: "position within judged pool (1 = best)", autorange: "reversed", zeroline: false, dtick: 5 },
  hovermode: "closest",
}, { responsive: true, displayModeBar: false });
</script>
</body>
</html>'''


if __name__ == "__main__":
    main()
