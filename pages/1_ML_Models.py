"""
GridLock — ML Architecture Deep Dive
Dedicated Streamlit page: two-model pipeline, data structures, feature tables, live charts.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import altair as alt


ss = st.session_state

# ── shared dark theme helper ──────────────────────────────────────────────────
def _dark(chart, height=300):
    return chart.configure(background="#0d1117").configure_view(
        strokeWidth=0
    ).configure_axis(
        gridColor="#1e2535", labelColor="#778", titleColor="#889",
        domainColor="#334",
    ).configure_title(color="#ccd", fontSize=13).properties(height=height)


# ── styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Page-wide dark base */
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0e1a !important;
    color: #ccd;
}
[data-testid="stSidebar"] { background: #0d1117 !important; }

/* Section cards */
.ml-card {
    background: #0d1117;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 22px;
    border: 1px solid #1e2535;
}
.ml-card h2 { color: #fff; margin-top: 0; }
.ml-card-accent-orange { border-left: 4px solid #ff6b35; }
.ml-card-accent-blue   { border-left: 4px solid #4488ff; }
.ml-card-accent-purple { border-left: 4px solid #cc88ff; }
.ml-card-accent-teal   { border-left: 4px solid #00d4aa; }

/* Feature category pills */
.feat-pill {
    display: inline-block;
    background: #1a2030;
    border: 1px solid #2a3555;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #99bbdd;
    margin: 3px 4px 3px 0;
}
/* Formula highlight */
.formula-big {
    background: #111827;
    border: 1px solid #2a3555;
    border-radius: 10px;
    padding: 18px 24px;
    margin: 12px 0;
    font-family: monospace;
    font-size: 1.05rem;
    text-align: center;
    line-height: 2;
}
/* Data schema table */
.schema-tbl { width: 100%; border-collapse: collapse; }
.schema-tbl th {
    background: #161d2e; color: #99aacc;
    font-size: 0.75rem; letter-spacing: 0.06em;
    text-transform: uppercase; padding: 8px 12px;
    text-align: left; border-bottom: 2px solid #2a3555;
}
.schema-tbl td {
    padding: 9px 12px; font-size: 0.85rem; color: #bbc;
    border-bottom: 1px solid #1a2030; vertical-align: top;
}
.schema-tbl tr:hover td { background: #111827; }
.col-name { color: #7bc8ff; font-family: monospace; font-size: 0.82rem; }
.col-type { color: #ffbb44; font-size: 0.78rem; }
.col-source { font-size: 0.78rem; color: #556; }
.col-m1 { text-align: center; }
.col-m2 { text-align: center; }
.used   { color: #00d4aa; font-weight: 700; }
.label  { color: #ff6b35; font-weight: 700; }
.new    { color: #cc88ff; font-weight: 700; font-size: 0.7rem;
          background: #2a1a40; border-radius: 4px; padding: 1px 6px; }

/* Phase banners */
.phase-banner {
    display: flex; align-items: center; gap: 14px;
    background: #111827; border-radius: 8px;
    padding: 12px 20px; margin: 14px 0;
    border: 1px solid #2a3555;
}
.phase-icon { font-size: 1.5rem; }
.phase-label { font-size: 0.65rem; color: #556; letter-spacing: 0.12em; text-transform: uppercase; }
.phase-title { color: #fff; font-size: 1rem; font-weight: 700; }
.phase-sub   { color: #778; font-size: 0.82rem; }

.insight-box {
    background: #0e1520;
    border-left: 3px solid #4488ff;
    border-radius: 6px;
    padding: 12px 18px;
    margin: 10px 0;
    font-size: 0.88rem;
    color: #99bbdd;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:8px">
  <span style="font-size:2.2rem">🧠</span>
  <div>
    <div style="font-size:1.7rem;font-weight:800;color:#fff;letter-spacing:0.02em">
      ML Architecture — Under The Hood
    </div>
    <div style="color:#556;font-size:0.9rem;margin-top:2px">
      GridLock · Flipkart GRID 2026 · Two-model LightGBM pipeline
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

calibrated = ss.get("calibrated", False)
if calibrated:
    st.success("Models loaded from session — live metrics and charts active below.", icon="✅")
else:
    st.info("Run Stage 4 (Calibrate AI) in the main app to unlock live charts and metrics on this page.", icon="ℹ️")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. FULL SYSTEM FLOWCHART (pure HTML/CSS — no external dependencies)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## The Two-Model Pipeline")
st.caption("How data flows from raw inputs through training to a deployment brief")

_PIPELINE_HTML = (
"""<!DOCTYPE html>
<html>
<head>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0e1a;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#ccd;padding:6px 4px}

/* ── DATA INPUTS ── */
.inputs{display:flex;gap:8px;margin-bottom:6px}
.inp{flex:1;background:#0d1320;border:1px solid #1e3055;border-radius:9px;padding:11px 13px}
.inp-t{color:#7bbcff;font-size:.83rem;font-weight:700;margin-bottom:3px}
.inp-s{color:#445;font-size:.71rem;line-height:1.4}
.tag{display:inline-block;margin-top:6px;border-radius:20px;padding:2px 9px;font-size:.62rem;font-weight:700}
.feeds{text-align:center;color:#2a3a50;font-size:.75rem;padding:4px 0;letter-spacing:.04em}

/* ── PHASE COLUMNS ── */
.phases{display:flex;gap:0;min-height:520px}
.tcol{flex:1;border:2px solid #332200;border-radius:10px 0 0 10px;overflow:hidden}
.pcol{flex:1;border:2px solid #1e3a5e;border-radius:0 10px 10px 0;overflow:hidden}
.ph-hd{padding:10px 14px;font-size:.66rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;border-bottom:2px solid}

/* ── STEPS ── */
.step{display:flex;gap:10px;align-items:flex-start;padding:12px 13px;border-bottom:1px solid #111827;transition:background .15s}
.step:last-child{border-bottom:none}
.num{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.8rem;font-weight:800;flex-shrink:0;margin-top:1px}
.ico{font-size:1.05rem;flex-shrink:0;margin-top:2px}
.stit{font-size:.84rem;font-weight:700;line-height:1.25;margin-bottom:4px}
.sdesc{font-size:.72rem;color:#5a6a7a;line-height:1.5}
.sdesc em{color:#8899aa;font-style:normal}

/* ── BRIDGE ── */
.bridge{flex:0 0 50px;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#080c14;border-top:2px solid #1e3055;border-bottom:2px solid #1e3055}
.bri{writing-mode:vertical-lr;transform:rotate(180deg);color:#4488ff;font-size:.64rem;font-weight:700;letter-spacing:.07em;background:#0d1a2e;border-radius:6px;padding:14px 5px;border:1px solid #1e3a5e;line-height:1.6}

/* ── STEP ACCENT COLOURS ── */
.s-f {border-left:3px solid #ff6b35}
.s-sp{border-left:3px solid #4488ff}
.s-m1{border-left:3px solid #aabbff;background:#0d1228}
.s-dt{border-left:3px solid #ff8833}
.s-vl{border-left:3px solid #00d4aa}
.s-ea{border-left:3px solid #cc88ff}
.s-m2{border-left:3px solid #cc88ff;background:#100d1e}
.s-ev{border-left:3px solid #ffaa44}
.s-bf{border-left:3px solid #4488ff}
.s-pd{border-left:3px solid #cc88ff;background:#100d1e}
.s-fm{border-left:3px solid #ff8833}
.s-si{border-left:3px solid #00d4aa}
.s-op{border-left:3px solid #4488ff}
.s-br{border-left:3px solid #00d4aa;background:#0a1e14}
</style>
</head>
<body>

<!-- DATA INPUTS ROW -->
<div class="inputs">
  <div class="inp">
    <div class="inp-t">&#128203; ASTraM Events</div>
    <div class="inp-s">8,173 traffic events &middot; Nov 2023&ndash;Apr 2024<br>Bengaluru Traffic Police real data</div>
    <span class="tag" style="background:#0a1a30;color:#7bbcff;border:1px solid #1e3055">Plane A &mdash; REAL DATA</span>
  </div>
  <div class="inp">
    <div class="inp-t">&#128230; Speed Panel</div>
    <div class="inp-s">Measured speed per road per 15-minute slot<br>150 days of readings per segment</div>
    <span class="tag" style="background:#1a0f00;color:#ff8833;border:1px solid #442200">Plane B &mdash; TomTom / Synthetic</span>
  </div>
  <div class="inp">
    <div class="inp-t">&#127758; Road Graph</div>
    <div class="inp-s">Every junction + road in 2&thinsp;km zone around<br>Chinnaswamy Stadium (OSMnx)</div>
    <span class="tag" style="background:#0a1a10;color:#44cc88;border:1px solid #1a4422">Plane C &mdash; OpenStreetMap</span>
  </div>
</div>
<div class="feeds">&#8595;&nbsp;&nbsp;All three data inputs are used by both phases below&nbsp;&nbsp;&#8595;</div>

<!-- TWO-PHASE COLUMNS -->
<div class="phases">

  <!-- ════ TRAINING PHASE ════ -->
  <div class="tcol">
    <div class="ph-hd" style="background:#1a1100;color:#ffaa44;border-color:#3a2a00">
      &#127919;&nbsp; TRAINING PHASE &mdash; Run once. Builds both models from historical data.
    </div>

    <div class="step s-f">
      <div class="num" style="background:#2a1100;color:#ff6b35">1</div>
      <div class="ico">&#128683;</div>
      <div>
        <div class="stit" style="color:#ff7744">Remove all event-day rows from training</div>
        <div class="sdesc">The baseline model must only learn from <em>calm, normal traffic days</em>. If it ever sees event congestion during training, it would think that&rsquo;s &ldquo;normal&rdquo; and the delta would shrink to near zero &mdash; hiding the event&rsquo;s true impact.</div>
      </div>
    </div>

    <div class="step s-sp">
      <div class="num" style="background:#0a1a30;color:#4488ff">2</div>
      <div class="ico">&#128197;</div>
      <div>
        <div class="stit" style="color:#7799ff">Split chronologically: first 80% train, last 20% test</div>
        <div class="sdesc">Always split by date, never randomly. A random split would let the model &ldquo;see&rdquo; future traffic patterns during training and produce <em>falsely optimistic</em> test scores that collapse in production.</div>
      </div>
    </div>

    <div class="step s-m1">
      <div class="num" style="background:#1a1a40;color:#aabbff">3</div>
      <div class="ico">&#129504;</div>
      <div>
        <div class="stit" style="color:#aabbff">Train Model 1 &mdash; LightGBM Baseline</div>
        <div class="sdesc">Learns the pattern: <em>road + time of day + weekday + rain &rarr; expected normal speed</em>. Output is <em>baseline_predicted</em> &mdash; the counterfactual &ldquo;what speed should this road be right now if nothing unusual happened?&rdquo;</div>
      </div>
    </div>

    <div class="step s-dt">
      <div class="num" style="background:#2a1500;color:#ff8833">4</div>
      <div class="ico">&#9889;</div>
      <div>
        <div class="stit" style="color:#ff9944">Compute &delta; = observed speed &minus; baseline predicted</div>
        <div class="sdesc">Done for <em>every single row</em>, event days and normal days alike. A negative delta (e.g. &minus;12 km/h) means traffic was slower than the baseline expected &mdash; and since the baseline already accounts for rush hour and rain, the delta is <em>purely the event&rsquo;s contribution</em>.</div>
      </div>
    </div>

    <div class="step s-vl">
      <div class="num" style="background:#0a2a1a;color:#00d4aa">5</div>
      <div class="ico">&#9989;</div>
      <div>
        <div class="stit" style="color:#00d4aa">Quality gate: non-event delta mean must be &asymp; 0</div>
        <div class="sdesc">On normal days, the model&rsquo;s prediction should match reality &mdash; so the delta should average near zero. If it doesn&rsquo;t, the baseline is systematically biased and <em>all downstream deltas are wrong</em>. This check must pass before continuing.</div>
      </div>
    </div>

    <div class="step s-ea">
      <div class="num" style="background:#220a40;color:#cc88ff">6</div>
      <div class="ico">&#127914;</div>
      <div>
        <div class="stit" style="color:#cc88ff">Filter event-active rows &amp; attach spatial features</div>
        <div class="sdesc">Keep only rows where <em>event_active = True</em>. Add: how many hops from the event junction, crowd size, event type. The measured deltas from Step 4 become the <em>training labels</em> for Model 2.</div>
      </div>
    </div>

    <div class="step s-m2">
      <div class="num" style="background:#220a40;color:#cc88ff">7</div>
      <div class="ico">&#128302;</div>
      <div>
        <div class="stit" style="color:#cc88ff">Train Model 2 &mdash; LightGBM Delta Predictor</div>
        <div class="sdesc">Learns: <em>hop distance + crowd + event type &rarr; how many km/h slower will this road be?</em> This model is what lets us <em>forecast future events before they happen</em> &mdash; no observed speed data needed at prediction time.</div>
      </div>
    </div>
  </div>

  <!-- ════ BRIDGE ════ -->
  <div class="bridge">
    <div class="bri">Model 2 output used in Planning Mode &#10230;</div>
  </div>

  <!-- ════ PLANNING PHASE ════ -->
  <div class="pcol">
    <div class="ph-hd" style="background:#0a1420;color:#4488ff;border-color:#1e3a5e">
      &#128203;&nbsp; PLANNING MODE &mdash; Run before each event. No observed speed data needed.
    </div>

    <div class="step s-ev">
      <div class="num" style="background:#1a1000;color:#ffaa44">8</div>
      <div class="ico">&#128221;</div>
      <div>
        <div class="stit" style="color:#ffaa44">Operator enters the upcoming event</div>
        <div class="sdesc">Inputs: event type (cricket / procession / rally), expected crowd size, location (junction), date and start time, duration. <em>No traffic data required</em> &mdash; the system runs 24 hours before the event.</div>
      </div>
    </div>

    <div class="step s-bf">
      <div class="num" style="background:#0a1a30;color:#4488ff">9</div>
      <div class="ico">&#127758;</div>
      <div>
        <div class="stit" style="color:#7799ff">BFS: measure hop distance for every road segment</div>
        <div class="sdesc">The system walks outward through the road network from the event junction (like ripples in a pond). Each road segment gets a &ldquo;hops from event&rdquo; number that represents its <em>network proximity</em> to the disruption source &mdash; not just straight-line distance.</div>
      </div>
    </div>

    <div class="step s-pd">
      <div class="num" style="background:#220a40;color:#cc88ff">10</div>
      <div class="ico">&#128302;</div>
      <div>
        <div class="stit" style="color:#cc88ff">Predict &delta; &plusmn; CI for every segment &times; every 15-min slot</div>
        <div class="sdesc">Model 2 runs for each road segment across every time bin during the event window. Output: predicted km/h slowdown <em>with a confidence interval</em> so operators see uncertainty, not just a point estimate.</div>
      </div>
    </div>

    <div class="step s-fm">
      <div class="num" style="background:#2a1200;color:#ff8833">11</div>
      <div class="ico">&#128506;</div>
      <div>
        <div class="stit" style="color:#ff9944">Build the congestion footprint map</div>
        <div class="sdesc">Roads ranked by <em>predicted impact &times; network centrality</em>. Shows exactly which roads, which 15-minute windows, and how severely each will be hit. This is the input for the optimizer &mdash; the &ldquo;what will break&rdquo; layer.</div>
      </div>
    </div>

    <div class="step s-si">
      <div class="num" style="background:#0a2a1a;color:#00d4aa">12</div>
      <div class="ico">&#9881;</div>
      <div>
        <div class="stit" style="color:#00d4aa">BPR Simulator measures each intervention</div>
        <div class="sdesc">For each option (officer at junction X, close road Y, open diversion Z), the simulator <em>runs the traffic model and measures</em> how much congestion is removed. Combinations are simulated together &mdash; <em>never added</em>, because interventions interact non-linearly.</div>
      </div>
    </div>

    <div class="step s-op">
      <div class="num" style="background:#0a1a30;color:#4488ff">13</div>
      <div class="ico">&#128208;</div>
      <div>
        <div class="stit" style="color:#7799ff">Greedy optimizer allocates resources under budget</div>
        <div class="sdesc">Assigns officers, barricades, and diversions within the available budget. Uses <em>submodular maximization</em> &mdash; mathematically guaranteed to reach at least 63% of the optimal solution, with a proof, not a claim.</div>
      </div>
    </div>

    <div class="step s-br">
      <div class="num" style="background:#0a2a1a;color:#00d4aa">14</div>
      <div class="ico">&#128203;</div>
      <div>
        <div class="stit" style="color:#00d4aa">Generate the deployment brief</div>
        <div class="sdesc">Plain-language plan: who goes where, by what time, and why. Every number (e.g. &ldquo;this barricade removes 8 km/h of congestion on MG Road from 18:00&ndash;20:15&rdquo;) is backed by a <em>simulation measurement</em> &mdash; not a guess.</div>
      </div>
    </div>
  </div>

</div>
</body>
</html>"""
)

components.html(_PIPELINE_HTML, height=760, scrolling=False)

st.markdown("""
<div class="insight-box">
<b>The core design principle:</b>
ML predicts the PROBLEM (how much congestion an event will cause).
Simulation PROVES the SOLUTION (how much each intervention removes).
Congestion is never invented — it is always either <b>measured</b> (δ = observed − baseline)
or <b>simulated</b> (SUMO). This is what makes the system honest enough to act on.
</div>
""", unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SHARED DATA SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## The Speed Panel — Shared Input Data Structure")
st.caption("Every column in the panel, what it means, and which model uses it")

st.markdown("""
<table class="schema-tbl">
<thead>
<tr>
  <th>Column</th><th>Type</th><th>Source</th><th>Description</th>
  <th style="color:#ffaa44">Model 1</th><th style="color:#cc88ff">Model 2</th>
</tr>
</thead>
<tbody>
<tr>
  <td class="col-name">segment_id</td>
  <td class="col-type">str</td>
  <td class="col-source">Road graph</td>
  <td>Unique edge ID: <code>node_u__node_v__key</code></td>
  <td class="col-m1"><span class="used">✓ key</span></td>
  <td class="col-m2"><span class="used">✓ key</span></td>
</tr>
<tr>
  <td class="col-name">date</td>
  <td class="col-type">date</td>
  <td class="col-source">Time axis</td>
  <td>Calendar date of the 15-min bin</td>
  <td class="col-m1"><span class="used">✓ split</span></td>
  <td class="col-m2"><span class="used">✓ split</span></td>
</tr>
<tr>
  <td class="col-name">hour</td>
  <td class="col-type">int 0–23</td>
  <td class="col-source">Time axis</td>
  <td>Hour of day (converted to sin/cos cyclical encoding)</td>
  <td class="col-m1"><span class="used">✓ feature</span></td>
  <td class="col-m2"><span class="used">✓ feature</span></td>
</tr>
<tr>
  <td class="col-name">hour_sin / hour_cos</td>
  <td class="col-type">float</td>
  <td class="col-source">Engineered</td>
  <td>Cyclical encoding: <code>sin(2π·hour/24)</code>  — 11pm and 1am are close</td>
  <td class="col-m1"><span class="used">✓ feature</span></td>
  <td class="col-m2"><span class="used">✓ feature</span></td>
</tr>
<tr>
  <td class="col-name">weekday_sin / weekday_cos</td>
  <td class="col-type">float</td>
  <td class="col-source">Engineered</td>
  <td>Cyclical day-of-week encoding — Sun and Mon are close</td>
  <td class="col-m1"><span class="used">✓ feature</span></td>
  <td class="col-m2"><span class="used">✓ feature</span></td>
</tr>
<tr>
  <td class="col-name">is_weekend</td>
  <td class="col-type">bool</td>
  <td class="col-source">Date</td>
  <td>Saturday or Sunday — demand pattern shifts significantly</td>
  <td class="col-m1"><span class="used">✓ feature</span></td>
  <td class="col-m2"><span class="used">✓ feature</span></td>
</tr>
<tr>
  <td class="col-name">is_rain</td>
  <td class="col-type">bool</td>
  <td class="col-source">Weather</td>
  <td>Rain flag for the date (synthetic now, real OWM in prod)</td>
  <td class="col-m1"><span class="used">✓ feature</span></td>
  <td class="col-m2"><span class="used">✓ feature</span></td>
</tr>
<tr>
  <td class="col-name">segment_code</td>
  <td class="col-type">int</td>
  <td class="col-source">Engineered</td>
  <td>Label-encoded road identity — captures each road's unique demand profile</td>
  <td class="col-m1"><span class="used">✓ feature</span></td>
  <td class="col-m2"><span class="used">✓ feature</span></td>
</tr>
<tr>
  <td class="col-name">lanes</td>
  <td class="col-type">int</td>
  <td class="col-source">Road graph</td>
  <td>Number of lanes from OSM <code>lanes</code> attribute</td>
  <td class="col-m1"><span class="used">✓ feature</span></td>
  <td class="col-m2"><span class="used">✓ feature</span></td>
</tr>
<tr>
  <td class="col-name">free_flow_speed</td>
  <td class="col-type">float km/h</td>
  <td class="col-source">Road graph</td>
  <td>Posted speed limit — ceiling on unimpeded travel</td>
  <td class="col-m1"><span class="used">✓ feature</span></td>
  <td class="col-m2"><span class="used">✓ feature</span></td>
</tr>
<tr>
  <td class="col-name">capacity</td>
  <td class="col-type">float</td>
  <td class="col-source">Engineered</td>
  <td><code>lanes × free_flow_speed</code> — surrogate for road throughput</td>
  <td class="col-m1"><span class="used">✓ feature</span></td>
  <td class="col-m2"><span class="used">✓ feature</span></td>
</tr>
<tr>
  <td class="col-name">observed_speed</td>
  <td class="col-type">float km/h</td>
  <td class="col-source">TomTom API</td>
  <td>Actual measured speed — REAL data, always. Never predicted or synthetic in prod.</td>
  <td class="col-m1"><span class="label">★ TARGET</span></td>
  <td class="col-m2"><span style="color:#556">—</span></td>
</tr>
<tr>
  <td class="col-name">is_event_day</td>
  <td class="col-type">bool</td>
  <td class="col-source">ASTraM join</td>
  <td>True if any ASTraM event is active in the zone on this date</td>
  <td class="col-m1"><span style="color:#ff6b35">✗ FILTER</span></td>
  <td class="col-m2"><span class="used">✓ key</span></td>
</tr>
<tr>
  <td class="col-name">event_active</td>
  <td class="col-type">bool</td>
  <td class="col-source">ASTraM join</td>
  <td>True if an event is active during this specific 15-min bin (narrower than is_event_day)</td>
  <td class="col-m1"><span style="color:#556">—</span></td>
  <td class="col-m2"><span class="used">✓ filter</span></td>
</tr>
<tr>
  <td class="col-name">event_type</td>
  <td class="col-type">str</td>
  <td class="col-source">ASTraM join</td>
  <td>Type: <code>cricket_match</code>, <code>political_gathering</code>, <code>procession</code>, etc.</td>
  <td class="col-m1"><span style="color:#556">—</span></td>
  <td class="col-m2"><span class="used">✓ feature</span></td>
</tr>
<tr style="background:#0e0e1a">
  <td class="col-name">baseline_predicted</td>
  <td class="col-type">float km/h</td>
  <td class="col-source"><b>Model 1 output</b></td>
  <td>What Model 1 predicts as normal speed here — used as context for Model 2</td>
  <td class="col-m1"><span class="label">↑ OUTPUT</span></td>
  <td class="col-m2"><span class="used">✓ feature <span class="new">NEW</span></span></td>
</tr>
<tr style="background:#0e0e1a">
  <td class="col-name">delta</td>
  <td class="col-type">float km/h</td>
  <td class="col-source"><b>Computed</b></td>
  <td><code>observed_speed − baseline_predicted</code> — the event's measured contribution</td>
  <td class="col-m1"><span style="color:#556">—</span></td>
  <td class="col-m2"><span class="label">★ TARGET</span></td>
</tr>
<tr style="background:#0e0e1a">
  <td class="col-name">hop_from_event</td>
  <td class="col-type">int</td>
  <td class="col-source"><b>BFS graph</b></td>
  <td>BFS hops from event junction to this segment — spatial decay driver</td>
  <td class="col-m1"><span style="color:#556">—</span></td>
  <td class="col-m2"><span class="used">✓ feature <span class="new">NEW</span></span></td>
</tr>
<tr style="background:#0e0e1a">
  <td class="col-name">hop_weight</td>
  <td class="col-type">float</td>
  <td class="col-source"><b>BFS graph</b></td>
  <td><code>capacity × decay(hop)</code> — capacity-adjusted spatial decay factor</td>
  <td class="col-m1"><span style="color:#556">—</span></td>
  <td class="col-m2"><span class="used">✓ feature <span class="new">NEW</span></span></td>
</tr>
<tr style="background:#0e0e1a">
  <td class="col-name">crowd</td>
  <td class="col-type">int</td>
  <td class="col-source"><b>ASTraM / prior</b></td>
  <td>Expected attendance — extracted from description or type-prior</td>
  <td class="col-m1"><span style="color:#556">—</span></td>
  <td class="col-m2"><span class="used">✓ feature <span class="new">NEW</span></span></td>
</tr>
<tr style="background:#0e0e1a">
  <td class="col-name">crowd_log</td>
  <td class="col-type">float</td>
  <td class="col-source"><b>Engineered</b></td>
  <td><code>log₁₀(crowd + 1)</code> — handles range from 500 to 50,000 without outlier distortion</td>
  <td class="col-m1"><span style="color:#556">—</span></td>
  <td class="col-m2"><span class="used">✓ feature <span class="new">NEW</span></span></td>
</tr>
<tr style="background:#0e0e1a">
  <td class="col-name">event_type_code</td>
  <td class="col-type">int</td>
  <td class="col-source"><b>Engineered</b></td>
  <td>Label-encoded event type — each type has a different congestion signature</td>
  <td class="col-m1"><span style="color:#556">—</span></td>
  <td class="col-m2"><span class="used">✓ feature <span class="new">NEW</span></span></td>
</tr>
</tbody>
</table>
""", unsafe_allow_html=True)

st.caption("Rows highlighted in dark blue are columns that only exist in the Model 2 training set — they're derived from Model 1's output and from the event + graph data.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODEL 1 — BASELINE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## Model 1 — Baseline (Normal Speed Predictor)")

col_m1a, col_m1b = st.columns([3, 2])

with col_m1a:
    st.markdown("""
<div class="ml-card ml-card-accent-orange">
<h2>🧠 LightGBM Baseline</h2>

<b style="color:#ffaa44">What it predicts:</b> Given any road segment at any time
of day, under any weather — <em>"what speed would this road be running at
if no event existed right now?"</em>
This is the <b>counterfactual</b>. It's what makes the delta measurement meaningful.

<br><br>

<b style="color:#ffaa44">Why we need it:</b> Without a counterfactual, we can't
isolate the event's contribution. Rush hour and rain both slow traffic too.
If we just measured speed on event days, we'd conflate all three causes.
The baseline separates them: it already knows about rush hour and rain,
so what's left over after subtracting it is <em>purely the event</em>.

<br><br>

<b style="color:#ffaa44">Training data:</b> Non-event rows only.
The <code>is_event_day = True</code> filter is the single most important
correctness guarantee in the entire system. If event traffic leaks in,
the model learns "slow roads are sometimes normal" → delta shrinks →
impact is undercounted → the whole downstream system is wrong.

</div>
""", unsafe_allow_html=True)

    st.markdown("### Features → Target")
    st.markdown("""
| Feature group | Columns | Role |
|---|---|---|
| **Road identity** | `segment_code` | Each road has its own demand pattern (main road vs side street) |
| **Time of day** | `hour_sin`, `hour_cos` | Rush hour, off-peak, night — cyclical so 11pm is close to 1am |
| **Day of week** | `weekday_sin`, `weekday_cos`, `is_weekend` | Weekday vs weekend traffic profiles are completely different |
| **Weather** | `is_rain` | Rain slows traffic — model learns this from non-event rainy days |
| **Road geometry** | `lanes`, `free_flow_speed`, `capacity` | A 4-lane arterial runs faster than a 2-lane side street |
| **→ Target** | `observed_speed` (km/h) | Real measured speed on this segment at this 15-min bin |
""")

with col_m1b:
    st.markdown("""
<div class="formula-big">
<span style="color:#aabbff">baseline_speed</span><br>
= LightGBM(<br>
&nbsp;&nbsp;<span style="color:#ffaa44">segment_code</span>,<br>
&nbsp;&nbsp;<span style="color:#44aaff">hour_sin, hour_cos</span>,<br>
&nbsp;&nbsp;<span style="color:#44aaff">weekday_sin, weekday_cos</span>,<br>
&nbsp;&nbsp;<span style="color:#44aaff">is_weekend</span>,<br>
&nbsp;&nbsp;<span style="color:#88ddaa">is_rain</span>,<br>
&nbsp;&nbsp;<span style="color:#ffcc44">lanes, ffs, capacity</span><br>
)<br><br>
<span style="color:#556;font-size:0.85rem">Trained on is_event_day = False rows ONLY</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("### Key numbers")
    bm = ss.get("bm")
    if bm and hasattr(bm, "metrics"):
        m = bm.metrics
        c1, c2 = st.columns(2)
        c1.metric("Test MAE", f"{m.get('test_mae', '—'):.2f} km/h")
        c2.metric("Train MAE", f"{m.get('train_mae', '—'):.2f} km/h")
        c3, c4 = st.columns(2)
        c3.metric("Training rows", f"{m.get('train_rows', 0):,}")
        c4.metric("Test rows", f"{m.get('test_rows', 0):,}")
    else:
        st.info("Run Stage 4 in the main app to see live metrics.")

    st.markdown("### Hyperparameters")
    st.markdown("""
| Setting | Value | Why |
|---|---|---|
| `n_estimators` | 300 | Enough trees to capture temporal patterns |
| `max_depth` | 6 | Prevents over-fitting to specific hour×segment combos |
| `learning_rate` | 0.05 | Slow-and-steady converges more generalisably |
| `min_child_samples` | 20 | Prevents leaf nodes from one rare segment |
| Train/test split | 80/20 | **Chronological** — first 80% of dates train, last 20% test |
""")

# ── Live feature importance chart ─────────────────────────────────────────────
if bm and hasattr(bm, "model") and bm.model is not None:
    with st.expander("📊 Live: Baseline Feature Importance + Validation Chart", expanded=True):
        fi_names = ["segment_code", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
                    "is_weekend", "is_rain", "lanes", "free_flow_speed", "capacity"]
        imps = list(bm.model.feature_importances_)
        n = min(len(fi_names), len(imps))
        fi_df = pd.DataFrame({"Feature": fi_names[:n], "Importance": imps[:n]})
        fi_df = fi_df.sort_values("Importance", ascending=False)

        panel_d = ss.get("panel_d")
        ch1, ch2 = st.columns(2)
        with ch1:
            chart = (alt.Chart(fi_df).mark_bar()
                     .encode(x=alt.X("Importance:Q"),
                             y=alt.Y("Feature:N", sort="-x"),
                             color=alt.Color("Importance:Q",
                                             scale=alt.Scale(scheme="oranges"), legend=None),
                             tooltip=["Feature:N", "Importance:Q"])
                     .properties(title="Baseline — Feature Importance"))
            st.altair_chart(_dark(chart, 280), use_container_width=True)
            st.caption("segment_code dominates: each road has its own baseline demand level")

        with ch2:
            if panel_d is not None:
                non_ev = panel_d[~panel_d["is_event_day"]]["delta"].sample(
                    min(20000, int((~panel_d["is_event_day"]).sum())), random_state=42)
                hist_df = pd.DataFrame({"delta": non_ev.values})
                h_chart = (alt.Chart(hist_df).mark_bar(color="#00d4aa", opacity=0.75)
                           .encode(x=alt.X("delta:Q", bin=alt.Bin(maxbins=80),
                                           title="δ on non-event rows"),
                                   y=alt.Y("count()", title="Count"))
                           .properties(title="Bias validation: non-event δ ≈ 0"))
                rule = (alt.Chart(pd.DataFrame({"x": [0]}))
                        .mark_rule(color="#ff6b35", strokeDash=[5, 3]).encode(x="x:Q"))
                st.altair_chart(_dark(h_chart + rule, 280), use_container_width=True)
                mean_d = panel_d[~panel_d["is_event_day"]]["delta"].mean()
                if abs(mean_d) < 0.5:
                    st.success(f"Non-event mean δ = {mean_d:+.4f} km/h ≈ 0  ✅  Baseline is unbiased")
                else:
                    st.error(f"Non-event mean δ = {mean_d:+.4f} km/h  ⚠️  Baseline has systematic bias")
            else:
                st.info("Run Stage 4 to see the validation histogram.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL 2 — DELTA PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## Model 2 — Delta Predictor (Event Impact Forecaster)")

col_m2a, col_m2b = st.columns([3, 2])

with col_m2a:
    st.markdown("""
<div class="ml-card ml-card-accent-purple">
<h2>🔮 LightGBM Delta Predictor</h2>

<b style="color:#cc88ff">What it predicts:</b> Given a <em>planned future event</em>
(type, crowd size, junction, timing), for every road segment in the zone —
"how many km/h slower will this segment be during the event, compared to normal?"

This is the <b>forecasting engine</b>. It runs <em>before</em> the event happens —
no observed speed needed. The output is a predicted Δ with confidence interval
per segment per 15-minute bin.

<br><br>

<b style="color:#cc88ff">What makes it different from Model 1:</b>
Model 1 is blind to events — it only knows road × time × weather.
Model 2 is the event specialist. It knows three things Model 1 doesn't:
<ul>
<li><b>Where</b> the event is (hop distance through the road graph)</li>
<li><b>How big</b> it is (crowd, log-scaled)</li>
<li><b>What kind</b> it is (cricket match vs procession affect different roads)</li>
</ul>

<br>

<b style="color:#cc88ff">Training data:</b> Event-active rows from the speed panel
(where <code>event_active = True</code>), after the delta has been computed by
subtracting Model 1's baseline prediction. The <b>measured delta is the label</b> —
so the model learns empirically from real events, not from assumptions.

</div>
""", unsafe_allow_html=True)

    st.markdown("### Feature groups — what Model 2 knows that Model 1 doesn't")

    feat_rows = [
        ("🎯 Spatial", "hop_from_event", "int 1–6",
         "BFS hop count from event junction to this segment through the road graph. "
         "Roads 1 hop away feel the full impact; roads 4+ hops away feel almost none. "
         "This is the primary spatial decay driver — no other team produces this.",
         True),
        ("🎯 Spatial", "hop_weight", "float",
         "capacity × exponential decay factor at that hop distance. "
         "A wide road 2 hops away gets a higher weight than a narrow road 1 hop away.",
         True),
        ("👥 Event", "crowd", "int",
         "Expected attendance. 5,000-person rally vs 50,000-person match causes very different congestion.",
         True),
        ("👥 Event", "crowd_log", "float",
         "log₁₀(crowd + 1). Handles the wide range (500→50,000) without large values "
         "dominating the model. Captures diminishing returns at extreme crowd sizes.",
         True),
        ("🎪 Event type", "event_type_code", "int",
         "Encoded event type. A cricket match saturates Cubbon Rd; a procession "
         "follows a route and saturates different roads. Each type has a different congestion signature.",
         True),
        ("📊 Context", "baseline_predicted", "float km/h",
         "Model 1's predicted normal speed for this segment at this time. "
         "A road already at 15 km/h normal speed is harder to slow further than one at 50 km/h. "
         "This lets Model 2 learn non-linear saturation effects.",
         True),
        ("⏰ Time", "hour_sin/cos, weekday", "float/bool",
         "Same cyclical time features as Model 1. Event impact varies by time: "
         "a match ending at 10pm causes worse congestion than one ending at 3pm.",
         False),
        ("🛣️ Road", "lanes, ffs, capacity, segment_code", "int/float",
         "Same road geometry as Model 1. High-capacity roads absorb more event traffic "
         "before slowing; low-capacity roads saturate immediately.",
         False),
        ("🌧️ Weather", "is_rain", "bool",
         "Rain + event is worse than either alone — Model 2 can capture this interaction.",
         False),
    ]

    for group, col_n, dtype, desc, is_new in feat_rows:
        new_badge = '<span class="new">NEW vs M1</span>' if is_new else ""
        st.markdown(
            f'<div style="display:flex;gap:10px;align-items:flex-start;'
            f'margin:6px 0;padding:10px 14px;background:#0d1117;'
            f'border-radius:8px;border:1px solid #1e2535">'
            f'<div style="min-width:110px;color:#778;font-size:0.78rem">{group}</div>'
            f'<div style="flex:1">'
            f'<code style="color:#cc88ff;font-size:0.85rem">{col_n}</code>'
            f' <span style="color:#556;font-size:0.78rem">{dtype}</span>'
            f' {new_badge}<br>'
            f'<span style="color:#889;font-size:0.82rem;line-height:1.5">{desc}</span>'
            f'</div></div>',
            unsafe_allow_html=True)

with col_m2b:
    st.markdown("""
<div class="formula-big">
<span style="color:#ff6b35">δ_predicted</span>&nbsp;±&nbsp;<span style="color:#cc88ff">CI</span><br>
= LightGBM(<br>
&nbsp;&nbsp;<span style="color:#cc88ff">hop_from_event</span>,<br>
&nbsp;&nbsp;<span style="color:#cc88ff">hop_weight</span>,<br>
&nbsp;&nbsp;<span style="color:#cc88ff">crowd, crowd_log</span>,<br>
&nbsp;&nbsp;<span style="color:#cc88ff">event_type_code</span>,<br>
&nbsp;&nbsp;<span style="color:#aabbff">baseline_predicted</span>,<br>
&nbsp;&nbsp;<span style="color:#44aaff">hour, weekday, is_rain</span>,<br>
&nbsp;&nbsp;<span style="color:#ffcc44">lanes, ffs, capacity</span><br>
)<br><br>
<span style="color:#556;font-size:0.85rem">Trained on event_active = True rows only<br>Label = measured δ from Stage 5</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("### Confidence intervals")
    st.markdown("""
Per-event-type residual standard deviation:
```python
resid = actual_delta − predicted_delta
resid_std = std(resid) by event_type

CI_lo = δ_pred − 1.5 × resid_std
CI_hi = δ_pred + 1.5 × resid_std
```
An event type with consistent historical residuals
(cricket matches always follow the same pattern)
→ narrow CI.

A rarely-seen or variable event type → wide CI,
reflecting genuine uncertainty.

**87% coverage** at 1.5σ for normally distributed
residuals.
""")

    dp = ss.get("dp")
    if dp and hasattr(dp, "_type_std") and dp._type_std:
        std_df = pd.DataFrame([
            {"Event type": k.replace("_", " ").title(),
             "Residual std (km/h)": round(v, 3),
             "CI width": f"±{1.5 * v:.2f} km/h"}
            for k, v in sorted(dp._type_std.items())
        ])
        st.dataframe(std_df, use_container_width=True, hide_index=True)
    else:
        st.markdown("_Run Stage 4 to see per-type CI widths._")

    st.markdown("### Key numbers")
    if dp and hasattr(dp, "metrics") and dp.metrics:
        m2 = dp.metrics
        c1, c2 = st.columns(2)
        c1.metric("Test MAE", f"{m2.get('test_mae_kmh', '—')} km/h")
        c2.metric("Train MAE", f"{m2.get('train_mae_kmh', '—')} km/h")
        c3, c4 = st.columns(2)
        c3.metric("Training rows", str(m2.get("train_rows", "—")))
        et = m2.get("event_types_seen", [])
        c4.metric("Event types", str(len(et)) if et else "—")
    else:
        st.info("Run Stage 4 in the main app to see live metrics.")

# ── Live Delta Predictor charts ───────────────────────────────────────────────
if dp and hasattr(dp, "model") and dp.model is not None:
    with st.expander("📊 Live: Delta Predictor Feature Importance + Delta Distribution", expanded=True):
        try:
            from delta_predictor import PREDICTOR_FEATURES
        except ImportError:
            PREDICTOR_FEATURES = []

        dp_ch1, dp_ch2 = st.columns(2)
        with dp_ch1:
            if PREDICTOR_FEATURES:
                dp_imps = list(dp.model.feature_importances_)
                n2 = min(len(PREDICTOR_FEATURES), len(dp_imps))
                dp_fi = pd.DataFrame({
                    "Feature": PREDICTOR_FEATURES[:n2],
                    "Importance": dp_imps[:n2]
                }).sort_values("Importance", ascending=False)

                feat_roles = {
                    "hop_from_event": "🎯 Spatial",
                    "hop_weight": "🎯 Spatial",
                    "crowd": "👥 Event",
                    "crowd_log": "👥 Event",
                    "event_type_code": "🎪 Event type",
                    "baseline_predicted": "📊 Context",
                    "capacity": "🛣️ Road",
                    "lanes": "🛣️ Road",
                    "free_flow_speed": "🛣️ Road",
                    "hour_sin": "⏰ Time",
                    "hour_cos": "⏰ Time",
                    "weekday_sin": "📅 Time",
                    "weekday_cos": "📅 Time",
                    "is_weekend": "📅 Time",
                    "is_rain": "🌧️ Weather",
                    "segment_code": "🛣️ Road",
                }
                dp_fi["Role"] = dp_fi["Feature"].map(feat_roles).fillna("")
                dp_chart = (alt.Chart(dp_fi).mark_bar()
                            .encode(x=alt.X("Importance:Q"),
                                    y=alt.Y("Feature:N", sort="-x"),
                                    color=alt.Color("Importance:Q",
                                                    scale=alt.Scale(scheme="purples"),
                                                    legend=None),
                                    tooltip=["Feature:N", "Role:N", "Importance:Q"])
                            .properties(title="Delta Predictor — Feature Importance"))
                st.altair_chart(_dark(dp_chart, 320), use_container_width=True)

        with dp_ch2:
            panel_d = ss.get("panel_d")
            if panel_d is not None and "delta" in panel_d.columns and "event_active" in panel_d.columns:
                ev_d = panel_d[panel_d["event_active"] == True]["delta"].dropna()
                ne_d = panel_d[panel_d["event_active"] == False]["delta"].sample(
                    min(len(ev_d) * 3, int((panel_d["event_active"] == False).sum())),
                    random_state=42).dropna()

                ev_df = pd.DataFrame({"delta": ev_d.values, "type": "Event-active rows"})
                ne_df = pd.DataFrame({"delta": ne_d.values, "type": "Non-event rows"})
                compare_df = pd.concat([ev_df, ne_df])

                layer = (alt.Chart(compare_df).mark_bar(opacity=0.65)
                         .encode(
                             x=alt.X("delta:Q", bin=alt.Bin(maxbins=60), title="δ (km/h)"),
                             y=alt.Y("count()", stack=None),
                             color=alt.Color("type:N",
                                             scale=alt.Scale(
                                                 domain=["Event-active rows", "Non-event rows"],
                                                 range=["#ff6b35", "#00d4aa"]),
                                             legend=alt.Legend(orient="bottom")))
                         .properties(title="δ distribution: event-active vs non-event"))
                st.altair_chart(_dark(layer, 320), use_container_width=True)
                ev_mean = ev_d.mean()
                st.caption(
                    f"Event-active mean δ: {ev_mean:+.2f} km/h  |  "
                    f"Non-event mean δ: {ne_d.mean():+.4f} km/h"
                )
            else:
                st.info("Run Stage 4 to see the delta distribution comparison.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 4b. SPATIAL PRIORITY SCORING — BFS HOPS × CENTRALITY
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## Spatial Priority Scoring — The Core Ranking Logic")
st.caption(
    "|δ| × hop_decay × centrality → which segments get officers first. "
    "Proximity matters, but network centrality can override it."
)

_SPATIAL_HTML = (
"""<!DOCTYPE html>
<html>
<head>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0e1a;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#ccd;padding:6px 4px}
.wrap{display:flex;gap:14px;align-items:flex-start}
.left{flex:0 0 54%}
.right{flex:1;display:flex;flex-direction:column;gap:10px}
.sec-lbl{color:#445;font-size:.61rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin-bottom:6px}
.card{background:#0d1117;border:1px solid #1e2535;border-radius:10px;padding:11px 13px}
/* Formula */
.frow{display:flex;align-items:baseline;gap:6px;margin-bottom:5px;font-size:.78rem}
.feq{color:#6677aa;font-size:.72rem;margin:2px 0}
.ft{color:#556;font-size:.69rem;line-height:1.5;margin-top:4px;border-top:1px solid #1e2535;padding-top:7px}
/* Ranking table */
.r-row{display:flex;align-items:center;gap:7px;padding:6px 0;border-bottom:1px solid #111827}
.r-row:last-child{border-bottom:none}
.medal{width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.7rem;font-weight:800;flex-shrink:0}
.rn{flex:1;font-size:.78rem;font-weight:600}
.rh{min-width:50px;text-align:center;border-radius:4px;padding:2px 5px;font-size:.65rem;font-weight:700}
.bw{flex:1;height:8px;background:#151d2e;border-radius:5px;overflow:hidden}
.bb{height:100%;border-radius:5px}
.rs{min-width:38px;text-align:right;font-size:.77rem;font-weight:700}
/* Insight */
.insight{border:1px solid #4422aa;background:#0c0d22;border-radius:8px;padding:11px 13px}
</style>
</head>
<body>
<div class="wrap">

<!-- ════ LEFT: NETWORK DIAGRAM ════ -->
<div class="left">
  <div class="sec-lbl">How |&delta;| &times; hop_decay &times; centrality combine into priority score</div>
  <div class="card" style="padding:10px 8px 6px">

  <svg viewBox="0 0 530 305" width="100%">

    <!-- BFS ZONE RINGS -->
    <circle cx="198" cy="188" r="88"  fill="#ff6b35" fill-opacity=".04"/>
    <circle cx="198" cy="188" r="165" fill="#ffaa44" fill-opacity=".028"/>
    <circle cx="198" cy="188" r="245" fill="#7788ff" fill-opacity=".018"/>
    <circle cx="198" cy="188" r="88"  fill="none" stroke="#ff6b35" stroke-opacity=".22" stroke-width="1.2" stroke-dasharray="4 5"/>
    <circle cx="198" cy="188" r="165" fill="none" stroke="#ffaa44" stroke-opacity=".16" stroke-width="1.2" stroke-dasharray="4 5"/>
    <circle cx="198" cy="188" r="245" fill="none" stroke="#7788ff" stroke-opacity=".13" stroke-width="1.2" stroke-dasharray="4 5"/>

    <!-- ZONE RING LABELS -->
    <text x="285" y="148" fill="#ff6b35" fill-opacity=".5" font-size="9.5" font-family="monospace">&#8592; HOP 1 zone  (decay &#215;0.70)</text>
    <text x="358" y="106" fill="#ffaa44" fill-opacity=".55" font-size="9.5" font-family="monospace">&#8592; HOP 2  (&#215;0.40)</text>
    <text x="435" y="68"  fill="#8899ff" fill-opacity=".6"  font-size="9.5" font-family="monospace">&#8592; HOP 3  (&#215;0.15)</text>

    <!-- ROAD CONNECTIONS (edges) -->
    <line x1="198" y1="188" x2="198" y2="103" stroke="#ff6b35" stroke-width="2.2" stroke-opacity=".65"/>
    <line x1="198" y1="188" x2="285" y2="188" stroke="#884433" stroke-width="1.5" stroke-opacity=".4"/>
    <line x1="198" y1="103" x2="198" y2="24"  stroke="#ffaa44" stroke-width="2"   stroke-opacity=".5"/>
    <line x1="285" y1="188" x2="355" y2="128" stroke="#ffaa44" stroke-width="2"   stroke-opacity=".45"/>
    <line x1="355" y1="128" x2="435" y2="52"  stroke="#cc88ff" stroke-width="2.5" stroke-opacity=".75" stroke-dasharray="6 3"/>

    <!-- ── NODE: CHINNASWAMY (event source) ── -->
    <circle cx="198" cy="188" r="26" fill="#991122" stroke="#ff3355" stroke-width="2.5"/>
    <text x="198" y="185" text-anchor="middle" fill="white" font-size="13" font-family="sans-serif">&#127903;</text>
    <text x="198" y="197" text-anchor="middle" fill="white" font-size="7" font-family="sans-serif">EVENT</text>
    <text x="198" y="228" text-anchor="middle" fill="#cc3355" font-size="10.5" font-family="sans-serif" font-weight="800">Chinnaswamy</text>
    <text x="198" y="241" text-anchor="middle" fill="#aa2244" font-size="8.5" font-family="sans-serif">Stadium &mdash; HOP 0</text>

    <!-- ── NODE: M.G. ROAD (hop 1, centrality 0.85, score 0.595 = RANK #1) ── -->
    <!-- Large node — high centrality, high priority -->
    <circle cx="198" cy="103" r="23" fill="#882211" stroke="#ff6b35" stroke-width="2.5"/>
    <text x="198" y="108" text-anchor="middle" fill="white" font-size="11" font-family="sans-serif" font-weight="800">#1</text>
    <text x="198" y="78"  text-anchor="middle" fill="#ff8855" font-size="11" font-family="sans-serif" font-weight="800">M.G. Road</text>
    <text x="198" y="90"  text-anchor="middle" fill="#ff6b35" font-size="8.5" font-family="monospace">HOP 1 &middot; c=0.85 &rarr; score=0.595</text>

    <!-- ── NODE: INFANTRY RD (hop 1, centrality 0.15, score 0.105 = RANK #5) ── -->
    <!-- TINY node — low centrality, low priority despite being hop 1! -->
    <circle cx="285" cy="188" r="8" fill="#443322" stroke="#886644" stroke-width="1.5"/>
    <text x="295" y="178" text-anchor="start" fill="#887755" font-size="10" font-family="sans-serif" font-weight="700">Infantry Rd</text>
    <text x="295" y="190" text-anchor="start" fill="#556" font-size="8" font-family="monospace">#5 &middot; HOP 1 &middot; c=0.15</text>
    <text x="295" y="201" text-anchor="start" fill="#444" font-size="8" font-family="monospace">score=0.105</text>

    <!-- ── NODE: RESIDENCY RD (hop 2, centrality 0.72, score 0.288 = RANK #2) ── -->
    <circle cx="198" cy="24" r="20" fill="#664400" stroke="#ffaa44" stroke-width="2"/>
    <text x="198" y="29" text-anchor="middle" fill="white" font-size="11" font-family="sans-serif" font-weight="800">#2</text>
    <text x="130" y="14" text-anchor="middle" fill="#ffaa44" font-size="10.5" font-family="sans-serif" font-weight="700">Residency Rd</text>
    <text x="130" y="27" text-anchor="middle" fill="#aa7722" font-size="8"   font-family="monospace">HOP 2 &middot; c=0.72 &rarr; 0.288</text>

    <!-- ── NODE: MUSEUM RD (hop 2, centrality 0.45, score 0.180 = RANK #3) ── -->
    <circle cx="355" cy="128" r="14" fill="#553300" stroke="#ccaa33" stroke-width="1.5"/>
    <text x="355" y="133" text-anchor="middle" fill="white" font-size="10" font-family="sans-serif" font-weight="800">#3</text>
    <text x="356" y="113" text-anchor="middle" fill="#ccaa33" font-size="10"  font-family="sans-serif" font-weight="700">Museum Rd</text>
    <text x="356" y="103" text-anchor="middle" fill="#886633" font-size="8"   font-family="monospace">HOP 2 &middot; c=0.45 &rarr; 0.180</text>

    <!-- ── NODE: BRIGADE RD (hop 3, centrality 0.90, score 0.135 = RANK #4) ── -->
    <!-- BIG node — highest centrality in zone! Outranks Infantry Rd despite being hop 3 -->
    <circle cx="435" cy="52" r="26"  fill="#220a44" stroke="#cc88ff" stroke-width="2.5"/>
    <circle cx="435" cy="52" r="34"  fill="none" stroke="#cc88ff" stroke-opacity=".2" stroke-width="1.5" stroke-dasharray="3 4"/>
    <text x="435" y="57" text-anchor="middle" fill="white" font-size="11" font-family="sans-serif" font-weight="800">#4</text>
    <text x="435" y="23" text-anchor="middle" fill="#cc88ff" font-size="11"  font-family="sans-serif" font-weight="800">Brigade Rd</text>
    <text x="435" y="10" text-anchor="middle" fill="#9966cc" font-size="8.5" font-family="monospace">HOP 3 &middot; c=0.90 &rarr; score=0.135</text>

    <!-- KEY INSIGHT callout box -->
    <rect x="360" y="155" width="163" height="72" rx="7" fill="#0e0d22" stroke="#4422aa" stroke-width="1.5"/>
    <text x="370" y="171" fill="#cc88ff" font-size="9" font-family="sans-serif" font-weight="800">&#9888; KEY INSIGHT</text>
    <text x="370" y="184" fill="#9999cc" font-size="8.5" font-family="sans-serif">Brigade Rd is 3 hops away</text>
    <text x="370" y="196" fill="#9999cc" font-size="8.5" font-family="sans-serif">but its centrality (0.90) is so</text>
    <text x="370" y="208" fill="#9999cc" font-size="8.5" font-family="sans-serif">high it beats Infantry Rd (hop 1).</text>
    <text x="370" y="220" fill="#cc88ff" font-size="8" font-family="sans-serif" font-weight="700">Closest &#8800; most critical.</text>

    <!-- Arrow from callout to Brigade node -->
    <line x1="435" y1="155" x2="435" y2="90" stroke="#cc88ff" stroke-opacity=".4" stroke-width="1" stroke-dasharray="3 3" marker-end="url(#arr)"/>
    <defs>
      <marker id="arr" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill="#cc88ff" fill-opacity=".4"/>
      </marker>
    </defs>

    <!-- LEGEND -->
    <rect x="4" y="274" width="350" height="28" rx="5" fill="#0d1117" fill-opacity=".9"/>
    <circle cx="18" cy="288" r="9" fill="#882211" stroke="#ff6b35" stroke-width="1.5"/>
    <text x="32" y="292" fill="#667" font-size="8" font-family="sans-serif">large node = high centrality</text>
    <circle cx="190" cy="288" r="4" fill="#443322" stroke="#886644" stroke-width="1.5"/>
    <text x="200" y="292" fill="#667" font-size="8" font-family="sans-serif">small = low centrality</text>
    <text x="270" y="292" fill="#666" font-size="8" font-family="sans-serif">  #N = priority rank</text>

  </svg>
  </div>
</div>

<!-- ════ RIGHT: FORMULA + RANKING ════ -->
<div class="right">
  <div>
    <div class="sec-lbl">Priority scoring formula — three components</div>
    <div class="card">
      <div class="feq" style="font-size:.75rem;margin-bottom:6px">
        <b style="color:#ff6b35">Step 1</b> &mdash; predicted delta (how bad is this road hit?):
      </div>
      <div style="font-size:.77rem;margin-bottom:8px;color:#8899aa;line-height:1.5">
        <b style="color:#ff6b35">|&delta;|</b> = |predicted_speed &minus; baseline_speed| in km/h.<br>
        Comes from LightGBM #2 (Delta Predictor). Larger = more congestion.
      </div>
      <div class="feq" style="font-size:.75rem;margin-bottom:6px">
        <b style="color:#ff9944">Step 2</b> &mdash; decay by hop number (how far from event?):
      </div>
      <div style="display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap">
        <span style="background:#2a1200;color:#ff6b35;border-radius:4px;padding:2px 7px;font-size:.72rem"><b>n=0</b> &times;1.00</span>
        <span style="background:#2a1200;color:#ff8833;border-radius:4px;padding:2px 7px;font-size:.72rem"><b>n=1</b> &times;0.70</span>
        <span style="background:#2a1800;color:#ffaa44;border-radius:4px;padding:2px 7px;font-size:.72rem"><b>n=2</b> &times;0.40</span>
        <span style="background:#1a1a40;color:#8899ff;border-radius:4px;padding:2px 7px;font-size:.72rem"><b>n=3</b> &times;0.15</span>
        <span style="background:#111;color:#445;border-radius:4px;padding:2px 7px;font-size:.72rem"><b>n=4+</b> &times;0.05</span>
      </div>
      <div class="feq" style="font-size:.75rem;margin-bottom:5px">
        <b style="color:#00d4aa">Step 3</b> &mdash; multiply by betweenness centrality (how critical is this road?):
      </div>
      <div style="font-size:.77rem;margin-bottom:8px;color:#8899aa;line-height:1.5">
        <b style="color:#00d4aa">centrality</b> = fraction of all network shortest paths passing through this road.<br>High centrality = critical bridge — congestion here cascades everywhere.
      </div>
      <div style="background:#111827;border-radius:6px;padding:9px 11px;font-size:.82rem">
        <b style="color:#cc88ff">priority_score</b>
        &nbsp;=&nbsp; <b style="color:#ff6b35">|&delta;|</b>
        &nbsp;&times;&nbsp; <b style="color:#ff8833">decay(n)</b>
        &nbsp;&times;&nbsp; <b style="color:#00d4aa">centrality</b>
      </div>
    </div>
  </div>

  <div>
    <div class="sec-lbl">Ranked by priority score &mdash; deployment order</div>
    <div class="card">
      <div class="r-row">
        <div class="medal" style="background:#882211;border:2px solid #ff6b35;color:white">1</div>
        <span class="rn">M.G. Road</span>
        <span class="rh" style="background:#2a1200;color:#ff6b35">HOP 1</span>
        <div class="bw"><div class="bb" style="width:100%;background:linear-gradient(90deg,#ff6b35,#ff9944)"></div></div>
        <span class="rs" style="color:#ff6b35">0.595</span>
      </div>
      <div class="r-row">
        <div class="medal" style="background:#664400;border:2px solid #ffaa44;color:white">2</div>
        <span class="rn">Residency Rd</span>
        <span class="rh" style="background:#2a1800;color:#ffaa44">HOP 2</span>
        <div class="bw"><div class="bb" style="width:48%;background:linear-gradient(90deg,#ffaa44,#ffcc44)"></div></div>
        <span class="rs" style="color:#ffaa44">0.288</span>
      </div>
      <div class="r-row">
        <div class="medal" style="background:#553300;border:2px solid #ccaa33;color:white">3</div>
        <span class="rn">Museum Rd</span>
        <span class="rh" style="background:#2a1800;color:#ccaa33">HOP 2</span>
        <div class="bw"><div class="bb" style="width:30%;background:linear-gradient(90deg,#ccaa33,#ddbb44)"></div></div>
        <span class="rs" style="color:#ccaa33">0.180</span>
      </div>
      <div class="r-row" style="background:#110a20;border-radius:6px;padding:6px 4px;margin:2px -4px">
        <div class="medal" style="background:#330a55;border:2px solid #cc88ff;color:white">4</div>
        <span class="rn" style="color:#cc88ff">Brigade Rd</span>
        <span class="rh" style="background:#1a0a30;color:#cc88ff">HOP 3</span>
        <div class="bw"><div class="bb" style="width:22.7%;background:linear-gradient(90deg,#cc88ff,#9966cc)"></div></div>
        <span class="rs" style="color:#cc88ff">0.135</span>
      </div>
      <div class="r-row" style="opacity:.5">
        <div class="medal" style="background:#222;border:2px solid #445;color:#667">5</div>
        <span class="rn" style="color:#556">Infantry Rd</span>
        <span class="rh" style="background:#1a1200;color:#554433">HOP 1</span>
        <div class="bw"><div class="bb" style="width:17.6%;background:#554433"></div></div>
        <span class="rs" style="color:#556">0.105</span>
      </div>
    </div>
  </div>

  <div class="insight">
    <div style="color:#cc88ff;font-size:.63rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px">Why this matters</div>
    <div style="font-size:.78rem;color:#8899bb;line-height:1.65">
      <b style="color:#cc88ff">Brigade Rd</b> is <b>3 hops</b> away yet ranks <b style="color:#cc88ff">#4</b>.<br>
      <b style="color:#556">Infantry Rd</b> is <b>1 hop</b> away but ranks <b style="color:#556">#5 (last)</b>.<br><br>
      Brigade Rd carries the <em>highest betweenness centrality</em> in the zone &mdash; almost every alternative route passes through it. Placing one officer there eases congestion across multiple upstream roads at once.<br><br>
      <span style="color:#445;font-size:.75rem">"Closest first" deployment would miss this entirely. GridLock scores it correctly.</span>
    </div>
  </div>
</div>

</div>
</body>
</html>"""
)

components.html(_SPATIAL_HTML, height=660, scrolling=False)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. HOW THE PREDICTION FLOWS TO OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## From Prediction to Deployment Brief")
st.caption("How the two model outputs combine to produce an actionable plan")

steps = [
    ("1", "#ffaa44", "🗓️ Event Announced",
     "A cricket match at Chinnaswamy is scheduled for Sunday 7pm. "
     "System receives: event_type=cricket_match, crowd=45000, junction=Chinnaswamy, start=19:00, duration=4h"),
    ("2", "#4488ff", "🌐 BFS Hop Distances",
     "Starting from the event junction, the system runs BFS through the road graph. "
     "Every segment gets a hop_from_event value. "
     "M.G. Road = 1 hop; Residency Rd = 2 hops; Ulsoor = 4 hops."),
    ("3", "#aabbff", "🧠 Baseline Query (Model 1)",
     "For each segment in the zone, Model 1 predicts what the normal Sunday 7–11pm speed would be. "
     "This becomes baseline_predicted — the context feature for Model 2."),
    ("4", "#cc88ff", "🔮 Delta Prediction (Model 2)",
     "Model 2 runs on every segment × every 15-min bin during the event window. "
     "Output: predicted δ ± CI per cell. "
     "M.G. Road 1-hop: -18 ± 4 km/h.  Residency 2-hop: -11 ± 5 km/h.  Ulsoor 4-hop: -2 ± 3 km/h."),
    ("5", "#ff6b35", "🗺️ Congestion Footprint",
     "The predicted deltas are mapped spatially and temporally. "
     "The system identifies the worst segments (hotspots), their peak windows, "
     "and ranks them by impact × centrality."),
    ("6", "#00d4aa", "⚙️ BPR Simulation",
     "For each candidate intervention (officer at junction X, barricade on road Y, diversion to road Z), "
     "the BPR simulator measures how much of the predicted congestion it removes. "
     "Combined effects are simulated together — not added — because they interact non-linearly."),
    ("7", "#88ddaa", "📐 Greedy Optimizer",
     "The optimizer allocates officers, barricades, and diversions under the resource budget. "
     "Greedy submodular maximisation guarantees ≥63% of the optimal solution. "
     "Output: specific assignments with measured impact values."),
    ("8", "#ccd", "📋 Mission Brief",
     "Natural-language deployment plan: \"Deploy 2 officers at Cubbon Rd junction by 18:30. "
     "Close MG Road northbound from 19:00. Divert via Museum Rd.\" "
     "Every number is backed by a simulation measurement."),
]

for num, color, title, detail in steps:
    st.markdown(
        f'<div style="display:flex;gap:14px;align-items:flex-start;'
        f'margin-bottom:10px;padding:14px 18px;background:#0d1117;'
        f'border-radius:10px;border-left:4px solid {color}">'
        f'<div style="min-width:32px;height:32px;border-radius:50%;'
        f'background:{color};color:#000;font-weight:800;font-size:0.9rem;'
        f'display:flex;align-items:center;justify-content:center">{num}</div>'
        f'<div><div style="color:{color};font-weight:700;font-size:0.95rem;'
        f'margin-bottom:4px">{title}</div>'
        f'<div style="color:#889;font-size:0.85rem;line-height:1.6">{detail}</div>'
        f'</div></div>',
        unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 6. KEY PRINCIPLES (judges will ask)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## Design Principles ")

principles = [
    ("P1", "#ff6b35",
     "Baseline trained on non-event data ONLY",
     "Never let event traffic leak into baseline training. The filter `is_event_day = False` "
     "is what makes the delta measurement honest. Without it, the model learns that 'roads can "
     "be slow for no reason', shrinks the delta, and undercounts event impact."),
    ("P2", "#4488ff",
     "Time-based train/test split — never random",
     "Random split leaks future traffic patterns into training. A model trained on Mon+Wed+Fri "
     "and tested on Tue 'already knows' what Tuesday looks like through adjacent data. "
     "Time-based split enforces that test = genuinely unseen future dates."),
    ("P3", "#00d4aa",
     "No fabricated congestion target",
     "Delta is either MEASURED (observed − baseline on real event days) or SIMULATED (SUMO). "
     "It is never invented by a regression model. This is what separates GridLock from teams "
     "that train a model to predict 'congestion score' they invented themselves."),
    ("P4", "#cc88ff",
     "Intervention effects measured by simulation, not decomposed from a prediction",
     "You cannot split a predicted congestion number into per-intervention shares mathematically. "
     "Instead: simulate baseline → add intervention 1 → measure removal. Add intervention 2 → "
     "measure additional removal. The difference IS the intervention's effect."),
    ("P5", "#ffaa44",
     "Combined effects are non-additive — simulate the combination",
     "Officer A at junction 1 removes 3 km/h. Officer B at junction 2 removes 2 km/h. "
     "Together they remove 4 km/h, not 5 — they share some of the same congestion pool. "
     "Only simulation captures this interaction. Summing individual effects is wrong."),
    ("P6", "#88ddaa",
     "Quantify uncertainty — CIs, not bare numbers",
     "Report predicted deltas with confidence intervals (e.g. −18 ± 4 km/h). "
     "A judge who asks 'how confident are you?' gets an honest statistical answer, "
     "not a point estimate that looks more precise than it is."),
]

p1, p2 = st.columns(2)
for i, (code, color, title, desc) in enumerate(principles):
    col = p1 if i % 2 == 0 else p2
    col.markdown(
        f'<div style="background:#0d1117;border-radius:10px;padding:16px 20px;'
        f'margin-bottom:12px;border-left:4px solid {color}">'
        f'<div style="color:{color};font-size:0.68rem;font-weight:700;'
        f'letter-spacing:.1em;margin-bottom:4px">{code}</div>'
        f'<div style="color:#eee;font-weight:700;font-size:0.92rem;margin-bottom:6px">{title}</div>'
        f'<div style="color:#778;font-size:0.83rem;line-height:1.55">{desc}</div>'
        f'</div>',
        unsafe_allow_html=True)

st.divider()
st.caption("GridLock · Flipkart GRID 2026 · Team Srikrit · ML Architecture Page")
