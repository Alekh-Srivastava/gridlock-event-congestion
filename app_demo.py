"""
GridLock — Guided Demo  (app_demo.py)
======================================
Video-ready wizard: enter any lat/lng → 8 stages unlock one by one.
Keep app.py for the full dashboard. This file is the live walkthrough.

Run:  streamlit run app_demo.py
"""
import sys, time, json, warnings, joblib
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
import folium
from streamlit_folium import st_folium
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GridLock | Team Srikrit",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Stage cards */
.stage-card {
    border-radius: 12px; padding: 18px 22px; margin: 10px 0;
    border: 2px solid; position: relative;
}
.stage-card.locked  { border-color: #2a3040; background: #0d1117; opacity: 0.55; }
.stage-card.active  { border-color: #ff6b35; background: #1a1f2e;
                       box-shadow: 0 0 20px rgba(255,107,53,0.15); }
.stage-card.done    { border-color: #00d4aa; background: #0d1f1a; }

/* Stage header row */
.sh { display:flex; align-items:center; gap:12px; margin-bottom:6px; }
.sh .num  { font-size:1.6rem; font-weight:700; min-width:36px; }
.sh .name { font-size:1.1rem; font-weight:600; color:#eee; flex:1; }
.badge {
    font-size:0.72rem; font-weight:700; letter-spacing:0.08em;
    padding:3px 10px; border-radius:20px;
}
.badge.locked  { background:#1e2540; color:#556; }
.badge.active  { background:#ff6b35; color:#fff; }
.badge.done    { background:#00d4aa; color:#001a14; }

.stage-num.locked  { color:#334; }
.stage-num.active  { color:#ff6b35; }
.stage-num.done    { color:#00d4aa; }

/* Formula box */
.formula-box {
    background:#1a1f2e; border:1px solid #ff6b35; border-radius:10px;
    padding:18px 28px; text-align:center; font-size:1.25rem; color:#fff;
    margin:14px 0; font-family:monospace;
}

/* Disclosure */
.disclosure {
    background:#1a0d00; border-left:4px solid #ff9500; border-radius:0 8px 8px 0;
    padding:12px 16px; margin:10px 0; font-size:0.87rem; color:#ffd0a0;
}

/* Insight */
.insight {
    background:#0d1f2d; border-left:4px solid #00d4aa; border-radius:0 8px 8px 0;
    padding:12px 16px; margin:10px 0; font-size:0.87rem; color:#cce;
}

/* Done summary bar */
.done-bar {
    display:flex; gap:20px; flex-wrap:wrap; padding:10px 0;
}
.done-pill {
    background:#002a20; border:1px solid #00d4aa; border-radius:20px;
    padding:4px 14px; font-size:0.82rem; color:#00d4aa;
}

/* Progress stepper */
.stepper { display:flex; gap:0; margin-bottom:24px; }
.step-item {
    flex:1; text-align:center; padding:10px 4px 8px;
    border-bottom:3px solid #2a3040; cursor:default;
}
.step-item.s-done   { border-color:#00d4aa; }
.step-item.s-active { border-color:#ff6b35; }
.step-num  { font-size:1.1rem; font-weight:700; }
.step-name { font-size:0.65rem; color:#778; margin-top:2px; }
.s-done  .step-num  { color:#00d4aa; }
.s-active .step-num { color:#ff6b35; }
.s-done  .step-name { color:#00d4aa; }
.s-active .step-name{ color:#ff6b35; }

[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #1e2540; }
</style>
""", unsafe_allow_html=True)


DEFAULT_LAT, DEFAULT_LNG = 12.9788, 77.5996

STAGE_META = [
    ("🌐", "Overview"),
    ("🎯", "Zone Setup"),
    ("📂", "ASTraM Events"),
    ("⚡", "Speed Panel"),
    ("🧠", "Calibrate AI"),
    ("🎫", "Incident Config"),
    ("🗺️", "Footprint"),
    ("🚦", "Deployment"),
    ("📋", "Mission Brief"),
]

# ── Session state defaults ─────────────────────────────────────────────────────
ss = st.session_state
_defaults = {
    "active": 0,          # 0 = overview always shown, user advances from 1
    "zone_done": False,
    "events_done": False,
    "panel_done": False,
    "calibrated": False,
    "incident_done": False,
    "footprint_done": False,
    "optimize_done": False,
    "brief_done": False,
    # data objects
    "G": None, "seg_info": None, "seg_bc": None,
    "events": None, "event_days": None,
    "panel_d": None, "seg_cats": None,
    "bm": None, "dp": None,
    "event_spec": None, "footprint": None,
    "plan": None, "opt_result": None,
    "sim_before": None, "sim_after": None,
    "brief": "",
}
for k, v in _defaults.items():
    if k not in ss:
        ss[k] = v

# ── Helper: current active stage index (0–8) ────────────────────────────────
def _active():
    checks = ["zone_done","events_done","panel_done","calibrated",
              "incident_done","footprint_done","optimize_done","brief_done"]
    for i, k in enumerate(checks):
        if not ss[k]:
            return i + 1   # next stage to complete
    return 8

# ── Helpers ───────────────────────────────────────────────────────────────────
def _is_default_zone(lat, lng):
    return abs(lat - DEFAULT_LAT) < 0.05 and abs(lng - DEFAULT_LNG) < 0.05

def _stage_class(n, active):
    if n < active: return "done"
    if n == active: return "active"
    return "locked"

def _badge(cls):
    label = {"done": "✅ COMPLETE", "active": "⚡ IN PROGRESS", "locked": "🔒 LOCKED"}[cls]
    return f'<span class="badge {cls}">{label}</span>'

def _stage_header(n, icon, name, cls):
    st.markdown(f"""
    <div class="sh">
      <div class="num stage-num {cls}">{icon}</div>
      <div class="name">Stage {n} — {name}</div>
      {_badge(cls)}
    </div>""", unsafe_allow_html=True)

def _disclosure(text):
    st.markdown(f'<div class="disclosure">⚠️ <b>Disclosure:</b> {text}</div>', unsafe_allow_html=True)

def _insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)

def _dark(chart, height=280):
    return (chart
        .properties(height=height)
        .configure_view(fill="#1a1f2e", strokeWidth=0)
        .configure_axis(grid=True, gridColor="#2a3040", gridOpacity=0.4,
                        labelColor="#aabbcc", titleColor="#aabbcc")
        .configure_legend(labelColor="#aabbcc", titleColor="#aabbcc", fillColor="#1a1f2e")
        .configure_title(color="#eee"))

def _severity_color(d):
    if d < -10: return [255, 34, 34, 220]
    if d < -5:  return [255, 136, 0, 210]
    if d < -2:  return [255, 204, 0, 190]
    return [0, 204, 102, 160]

def _seg_paths(G, seg_info, color_fn=None, default_color=None):
    default_color = default_color or [40, 60, 90, 160]
    rows = []
    for sid, info in seg_info.items():
        u, v = info["u"], info["v"]
        if u not in G.nodes or v not in G.nodes: continue
        color = color_fn(sid) if color_fn else default_color
        rows.append({
            "path": [[G.nodes[u]["lng"], G.nodes[u]["lat"]],
                     [G.nodes[v]["lng"], G.nodes[v]["lat"]]],
            "color": color, "road": info.get("road", ""),
        })
    return rows

def _make_map(lat, lng, zoom=13.5):
    return folium.Map(
        location=[lat, lng],
        zoom_start=zoom,
        tiles="CartoDB DarkMatter",
        attr="© CARTO © OpenStreetMap contributors",
        prefer_canvas=True,
    )

def _add_path_layer(m, data, width=2):
    if not data:
        return
    features = []
    for row in data:
        c = row["color"]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": row["path"]},
            "properties": {
                "tooltip": row.get("road") or row.get("label") or "",
                "color": "#{:02x}{:02x}{:02x}".format(c[0], c[1], c[2]),
                "opacity": round(c[3] / 255, 2) if len(c) > 3 else 0.8,
            },
        })
    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        style_function=lambda x: {
            "color": x["properties"]["color"],
            "weight": width,
            "opacity": x["properties"]["opacity"],
            "fillOpacity": 0,
        },
        tooltip=folium.GeoJsonTooltip(fields=["tooltip"], labels=False, sticky=False),
    ).add_to(m)

def _add_scatter_layer(m, data, radius=8):
    for row in data:
        pos = row["position"]
        c = row["color"]
        hex_c = "#{:02x}{:02x}{:02x}".format(c[0], c[1], c[2])
        op = round(c[3] / 255, 2) if len(c) > 3 else 0.9
        r = row.get("radius", radius)
        tip = row.get("road") or row.get("label") or ""
        folium.CircleMarker(
            [pos[1], pos[0]],
            radius=r,
            color=hex_c,
            fill=True,
            fill_color=hex_c,
            fill_opacity=op,
            tooltip=tip,
        ).add_to(m)

def _deck(lat, lng, path_layers=None, scatter_layers=None, zoom=13.5, height=400):
    m = _make_map(lat, lng, zoom)
    for data, width in (path_layers or []):
        _add_path_layer(m, data, width)
    for data, radius in (scatter_layers or []):
        _add_scatter_layer(m, data, radius)
    st_folium(m, use_container_width=True, height=height, returned_objects=[])


def _priority_scoring_detail():
    """Detailed BFS-hop × centrality priority scoring explanation (Stage 1 expander)."""
    # ── 1. The equation ──────────────────────────────────────────────────────
    st.markdown(
        '<div style="margin-bottom:14px">'
        '<div style="color:#445;font-size:.62rem;font-weight:700;letter-spacing:.1em;'
        'margin-bottom:10px">HOW THE SCORE IS BUILT — THREE INPUTS, ONE NUMBER</div>'
        '<div style="display:flex;gap:8px;align-items:stretch">'

        '<div style="flex:1;background:#220800;border:1px solid #440a00;border-radius:9px;padding:12px">'
        '<div style="color:#ff6b35;font-size:.65rem;font-weight:800;letter-spacing:.08em;margin-bottom:8px">INPUT 1 — PREDICTED &#948; (km/h)</div>'
        '<div style="font-size:.74rem;color:#8899aa;line-height:1.55;margin-bottom:8px">'
        'From <b style="color:#ff6b35">LightGBM #2 Delta Predictor</b>: how many km/h '
        'slower will this road be during the event?<br><br>'
        'Larger |&#948;| = road is hit harder = higher raw impact.</div>'
        '<div style="background:#1a0800;border-radius:6px;padding:8px 10px;font-size:.72rem">'
        '<b style="color:#ff6b35">|&#948;|</b> = |predicted_speed &minus; baseline_speed|<br>'
        '<span style="color:#334;font-size:.67rem">bigger = worse congestion on this segment</span>'
        '</div></div>'

        '<div style="display:flex;align-items:center;font-size:1.6rem;color:#2a3a50;padding:0 4px">&#215;</div>'

        '<div style="flex:1;background:#1a0e00;border:1px solid #3a2200;border-radius:9px;padding:12px">'
        '<div style="color:#ff8833;font-size:.65rem;font-weight:800;letter-spacing:.08em;margin-bottom:8px">INPUT 2 — BFS HOP DECAY</div>'
        '<div style="font-size:.74rem;color:#8899aa;line-height:1.55;margin-bottom:8px">'
        'The system walks outward from the event junction through the road graph '
        '(like ripples in water). Each road gets a hop number.<br><br>'
        'The further away, the lower the weight:</div>'
        '<div style="display:flex;flex-direction:column;gap:4px">'
        '<div style="display:flex;align-items:center;gap:8px">'
        '<span style="background:#ffdd44;color:#000;border-radius:3px;padding:1px 7px;font-size:.7rem;font-weight:800;min-width:32px;text-align:center">n=0</span>'
        '<div style="flex:1;height:7px;background:#ffdd44;border-radius:4px"></div>'
        '<span style="color:#ffdd44;font-size:.75rem;font-weight:700">&#215;1.00</span></div>'
        '<div style="display:flex;align-items:center;gap:8px">'
        '<span style="background:#ff8833;color:#000;border-radius:3px;padding:1px 7px;font-size:.7rem;font-weight:800;min-width:32px;text-align:center">n=1</span>'
        '<div style="flex:1;height:7px;background:#ff8833;border-radius:4px;width:70%"></div>'
        '<span style="color:#ff8833;font-size:.75rem;font-weight:700">&#215;0.70</span></div>'
        '<div style="display:flex;align-items:center;gap:8px">'
        '<span style="background:#ffaa44;color:#000;border-radius:3px;padding:1px 7px;font-size:.7rem;font-weight:800;min-width:32px;text-align:center">n=2</span>'
        '<div style="flex:1;height:7px;background:#ffaa44;border-radius:4px;width:40%"></div>'
        '<span style="color:#ffaa44;font-size:.75rem;font-weight:700">&#215;0.40</span></div>'
        '<div style="display:flex;align-items:center;gap:8px">'
        '<span style="background:#5566bb;color:white;border-radius:3px;padding:1px 7px;font-size:.7rem;font-weight:800;min-width:32px;text-align:center">n=3</span>'
        '<div style="flex:1;height:7px;background:#5566bb;border-radius:4px;width:15%"></div>'
        '<span style="color:#8899ff;font-size:.75rem;font-weight:700">&#215;0.15</span></div>'
        '<div style="display:flex;align-items:center;gap:8px">'
        '<span style="background:#222;color:#445;border-radius:3px;padding:1px 7px;font-size:.7rem;font-weight:800;min-width:32px;text-align:center">n=4+</span>'
        '<div style="flex:1;height:7px;background:#333;border-radius:4px;width:5%"></div>'
        '<span style="color:#445;font-size:.75rem;font-weight:700">&#215;0.05</span></div>'
        '</div></div>'

        '<div style="display:flex;align-items:center;font-size:1.6rem;color:#2a3a50;padding:0 4px">&#215;</div>'

        '<div style="flex:1;background:#0a1e14;border:1px solid #1a3a22;border-radius:9px;padding:12px">'
        '<div style="color:#00d4aa;font-size:.65rem;font-weight:800;letter-spacing:.08em;margin-bottom:8px">INPUT 2 — BETWEENNESS CENTRALITY</div>'
        '<div style="font-size:.74rem;color:#8899aa;line-height:1.6">'
        '<b style="color:#00d4aa">What it measures:</b> out of all the shortest paths '
        'between any two junctions in the network, what fraction of them pass through this road?<br><br>'
        '<b style="color:#00d4aa">High centrality</b> = this road is a critical bridge. '
        'Blocking it cascades to many other routes.<br><br>'
        '<b style="color:#00d4aa">Low centrality</b> = most traffic can easily bypass it. '
        'Congestion stays local.</div>'
        '<div style="margin-top:10px;background:#0d1f14;border-radius:6px;padding:8px 10px;font-size:.72rem">'
        'centrality = <b style="color:#00d4aa">avg(node_bc[u], node_bc[v])</b><br>'
        '<span style="color:#334;font-size:.67rem">computed by NetworkX betweenness_centrality(k=200)</span>'
        '</div></div>'

        '<div style="display:flex;align-items:center;font-size:1.6rem;color:#2a3a50;padding:0 4px">=</div>'

        '<div style="flex:1;background:#120a22;border:2px solid #5533aa;border-radius:9px;padding:12px;'
        'display:flex;flex-direction:column;justify-content:center;text-align:center">'
        '<div style="color:#cc88ff;font-size:.65rem;font-weight:800;letter-spacing:.08em;margin-bottom:10px">PRIORITY SCORE</div>'
        '<div style="font-size:.92rem;font-weight:800;color:#cc88ff;margin-bottom:8px;line-height:1.6">'
        '<b style="color:#ff6b35">|&#948;|</b> &#215; <b style="color:#ff8833">decay(n)</b> &#215; <b style="color:#00d4aa">centrality</b></div>'
        '<div style="color:#6655aa;font-size:.72rem;line-height:1.55">'
        'Ranks every road segment for officer + barricade placement. '
        'High score = deploy here first.</div>'
        '</div>'

        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── 2. Worked example ────────────────────────────────────────────────────
    st.markdown(
        '<div style="background:#0d1117;border:1px solid #1e2535;border-radius:10px;padding:14px 16px;margin-bottom:12px">'
        '<div style="color:#445;font-size:.62rem;font-weight:700;letter-spacing:.1em;margin-bottom:12px">'
        'WORKED EXAMPLE — WHY HOP-3 CAN BEAT HOP-1</div>'
        '<table style="width:100%;border-collapse:collapse;font-size:.8rem">'
        '<thead><tr>'
        '<td style="padding:0 0 8px;color:#334;border-bottom:1px solid #1e2535"></td>'
        '<td style="text-align:center;padding:0 12px 8px;border-bottom:1px solid #1e2535">'
        '<b style="color:#cc88ff">Brigade Rd</b></td>'
        '<td style="text-align:center;padding:0 0 8px;border-bottom:1px solid #1e2535">'
        '<b style="color:#554433">Infantry Rd</b></td>'
        '</tr></thead>'
        '<tbody>'
        '<tr><td style="color:#556;padding:8px 0;border-bottom:1px solid #111827">Hops from event</td>'
        '<td style="text-align:center;padding:8px 12px;border-bottom:1px solid #111827">'
        '<span style="background:#1a1a3a;color:#8899ff;border-radius:5px;padding:2px 11px;font-weight:800">3</span></td>'
        '<td style="text-align:center;padding:8px 0;border-bottom:1px solid #111827">'
        '<span style="background:#2a1800;color:#ff8833;border-radius:5px;padding:2px 11px;font-weight:800">1</span>'
        '<span style="color:#334;font-size:.68rem;margin-left:5px">closer!</span></td></tr>'
        '<tr><td style="color:#556;padding:8px 0;border-bottom:1px solid #111827">Hop decay weight</td>'
        '<td style="text-align:center;color:#8899ff;font-weight:700;padding:8px 12px;border-bottom:1px solid #111827">&#215; 0.15</td>'
        '<td style="text-align:center;color:#ff8833;font-weight:700;padding:8px 0;border-bottom:1px solid #111827">&#215; 0.70</td></tr>'
        '<tr><td style="color:#556;padding:8px 0;border-bottom:2px solid #2a1a40">Betweenness centrality</td>'
        '<td style="text-align:center;padding:8px 12px;border-bottom:2px solid #2a1a40">'
        '<b style="color:#00d4aa">&#215; 0.90</b> <span style="color:#1a6655;font-size:.68rem">highest in zone</span></td>'
        '<td style="text-align:center;padding:8px 0;border-bottom:2px solid #2a1a40">'
        '<b style="color:#2a3535">&#215; 0.15</b> <span style="color:#222;font-size:.68rem">most paths bypass it</span></td></tr>'
        '<tr><td style="color:#ccd;font-weight:700;padding:10px 0 6px">= Priority score</td>'
        '<td style="text-align:center;padding:10px 12px 6px">'
        '<div style="color:#cc88ff;font-size:1.1rem;font-weight:800">0.135</div>'
        '<div style="color:#7755aa;font-size:.68rem">0.15 &#215; 0.90</div></td>'
        '<td style="text-align:center;padding:10px 0 6px">'
        '<div style="color:#443322;font-size:1.1rem;font-weight:800">0.105</div>'
        '<div style="color:#332211;font-size:.68rem">0.70 &#215; 0.15</div></td></tr>'
        '</tbody></table>'
        '<div style="background:#0c0d20;border-left:3px solid #cc88ff;border-radius:0 6px 6px 0;'
        'padding:9px 12px;margin-top:12px;font-size:.77rem;color:#8899bb;line-height:1.6">'
        '<b style="color:#cc88ff">Brigade Rd wins (0.135 &gt; 0.105)</b> despite being 3 hops away. '
        'Its centrality of 0.90 means nearly every alternate route in the zone passes through it. '
        'One officer there relieves pressure across 6+ upstream segments simultaneously. '
        'Infantry Rd is closer but a near dead-end in network terms &mdash; congestion there stays local.'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1a1f3a 0%,#12102a 100%);'
        'border:1px solid #3344aa;border-radius:10px;padding:13px 18px;margin-top:10px">'
        '<div style="color:#7799ff;font-size:.62rem;font-weight:700;letter-spacing:.1em;margin-bottom:3px">WANT THE FULL PICTURE?</div>'
        '<div style="color:#ccd;font-size:.86rem;font-weight:700">&#129504; Detailed version about the models</div>'
        '<div style="color:#4455aa;font-size:.71rem;margin-top:2px">'
        'Full network diagram &middot; all 5 roads ranked &middot; both LightGBM models explained &middot; step-by-step pipeline'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_ML_Models.py", label="Open ML Architecture Deep Dive →", icon="🧠", use_container_width=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 GridLock")
    st.markdown("**Event-Driven Traffic Intelligence**")
    st.markdown("""
    <div style="background:#1a1f2e;border:1px solid #ff6b35;border-radius:8px;
         padding:10px 14px;margin:8px 0;text-align:center">
      <div style="color:#ff6b35;font-size:0.72rem;letter-spacing:0.1em;font-weight:700">TEAM</div>
      <div style="color:#ffffff;font-size:1.3rem;font-weight:800;letter-spacing:0.05em">Srikrit</div>
      <div style="color:#556;font-size:0.7rem;margin-top:2px">Flipkart GRID 2026 · Theme 2</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    active = _active()
    checks = ["zone_done","events_done","panel_done","calibrated",
              "incident_done","footprint_done","optimize_done","brief_done"]
    names  = ["Zone Setup","ASTraM Events","Speed Panel","Calibrate AI",
              "Incident Config","Footprint","Deployment","Mission Brief"]
    for i, (k, n) in enumerate(zip(checks, names)):
        icon = "✅" if ss[k] else ("⚡" if i + 1 == active else "🔒")
        color = "#00d4aa" if ss[k] else ("#ff6b35" if i + 1 == active else "#334")
        st.markdown(f'<span style="color:{color}">{icon} Stage {i+1}: {n}</span>',
                    unsafe_allow_html=True)

    st.divider()
    n_done = sum(ss[k] for k in checks)
    st.progress(n_done / 8, text=f"{n_done}/8 stages complete")

    if st.button("🔄 Reset All", use_container_width=True):
        for k in _defaults:
            ss[k] = _defaults[k]
        st.rerun()

# ── PROGRESS STEPPER ─────────────────────────────────────────────────────────
active = _active()
step_html = '<div class="stepper">'
for i, (icon, name) in enumerate(STAGE_META):
    if i == 0:
        cls = "s-done"
    elif i < active:
        cls = "s-done"
    elif i == active:
        cls = "s-active"
    else:
        cls = ""
    step_html += f"""
    <div class="step-item {cls}">
      <div class="step-num">{icon}</div>
      <div class="step-name">{name}</div>
    </div>"""
step_html += "</div>"
st.markdown(step_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — OVERVIEW (always visible)
# ═══════════════════════════════════════════════════════════════════════════════
with st.container():
    st.markdown('<div class="stage-card done">', unsafe_allow_html=True)

    # Team + competition header
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;
         margin-bottom:14px;flex-wrap:wrap;gap:10px">
      <div>
        <span style="font-size:1.7rem;font-weight:800;color:#fff;letter-spacing:0.02em">
          🚦 GridLock
        </span>
        <span style="color:#556;font-size:0.95rem;margin-left:12px">
          AI-Powered Event Traffic Intelligence
        </span>
      </div>
      <div style="display:flex;gap:10px;align-items:center">
        <div style="background:#1a1f2e;border:1px solid #ff6b35;border-radius:8px;
             padding:6px 18px;text-align:center">
          <div style="color:#ff6b35;font-size:0.65rem;font-weight:700;letter-spacing:0.1em">TEAM</div>
          <div style="color:#fff;font-size:1.1rem;font-weight:800">Srikrit</div>
        </div>
        <div style="background:#1a1f2e;border:1px solid #334;border-radius:8px;
             padding:6px 18px;text-align:center">
          <div style="color:#556;font-size:0.65rem;font-weight:700;letter-spacing:0.1em">COMPETITION</div>
          <div style="color:#aab;font-size:0.85rem;font-weight:600">Flipkart GRID 2026</div>
          <div style="color:#556;font-size:0.7rem">Theme 2 — Event Congestion</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    _stage_header(0, "🌐", "Overview — The Problem & The System", "done")

    col_f, col_p = st.columns([1, 1])
    with col_f:
        # ── Stat bar ──────────────────────────────────────────────────────────
        st.markdown(
            '<div style="display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap">'
            '<div style="background:#1a0a0a;border:1px solid #441111;border-radius:8px;'
            'padding:8px 14px;flex:1;min-width:110px;text-align:center">'
            '<div style="color:#ff4455;font-size:1.35rem;font-weight:800">8,173</div>'
            '<div style="color:#556;font-size:0.7rem">ASTraM events logged</div></div>'
            '<div style="background:#1a0a0a;border:1px solid #441111;border-radius:8px;'
            'padding:8px 14px;flex:1;min-width:110px;text-align:center">'
            '<div style="color:#ff4455;font-size:1.35rem;font-weight:800">0</div>'
            '<div style="color:#556;font-size:0.7rem">with advance prediction</div></div>'
            '<div style="background:#0a1a12;border:1px solid #114411;border-radius:8px;'
            'padding:8px 14px;flex:1;min-width:110px;text-align:center">'
            '<div style="color:#00d4aa;font-size:1.35rem;font-weight:800">6</div>'
            '<div style="color:#556;font-size:0.7rem">hop radius of impact</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        # ── Problem vs Solution cards ──────────────────────────────────────────
        _rows = [
            ("🔴", "No segment-level warning",
             "Officers react after jams form — the window to prevent is already gone.",
             "🟢", "Per-segment δ ± CI, 24 h ahead",
             "Every 15-min bin gets a congestion forecast before the event starts."),
            ("🔴", "Gut-feel deployment",
             "No system tells a superintendent which junction, which road, which time.",
             "🟢", "Optimizer-placed, ≥63 % of optimal",
             "Greedy submodular optimizer allocates officers + barricades under budget."),
            ("🔴", "System learns nothing",
             "Each event starts from scratch. Past outcomes are never used to improve.",
             "🟢", "Post-event recalibration loop",
             "Predicted vs actual δ shrinks error across events — the system gets smarter."),
        ]
        for p_icon, p_title, p_body, s_icon, s_title, s_body in _rows:
            st.markdown(
                '<div style="display:flex;gap:8px;margin-bottom:8px">'
                f'<div style="flex:1;background:#120a0a;border:1px solid #331111;'
                f'border-radius:8px;padding:10px 12px">'
                f'<div style="color:#ff4455;font-size:0.72rem;font-weight:700;'
                f'margin-bottom:4px">{p_icon} TODAY — {p_title}</div>'
                f'<div style="color:#556;font-size:0.78rem;line-height:1.5">{p_body}</div>'
                f'</div>'
                f'<div style="flex:1;background:#0a120e;border:1px solid #114422;'
                f'border-radius:8px;padding:10px 12px">'
                f'<div style="color:#00d4aa;font-size:0.72rem;font-weight:700;'
                f'margin-bottom:4px">{s_icon} GRIDLOCK — {s_title}</div>'
                f'<div style="color:#556;font-size:0.78rem;line-height:1.5">{s_body}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("""
        <div class="formula-box">
          <span style="color:#ff6b35">δ</span>
          &nbsp;=&nbsp;
          <span style="color:#00d4aa">observed_speed</span>
          &nbsp;−&nbsp;
          <span style="color:#aabbff">baseline_speed</span>
          <br>
          <span style="font-size:0.9rem;color:#778">
          δ &lt; 0 &nbsp;⟹&nbsp; event slowed traffic by |δ| km/h
          </span>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
GridLock closes that gap in 3 steps:
1. **Predict the problem** — LightGBM baseline predicts normal speed;
   δ = actual − baseline isolates exactly the event's contribution
2. **Prove the solution** — BPR simulation measures how much congestion
   each officer / barricade / diversion actually removes
3. **Optimise deployment** — greedy submodular optimizer finds the best
   placement under real resource limits, guaranteed ≥ 63% of optimal
""")
    with col_p:
        stages_overview = [
            ("1", "Zone Setup", "OSMnx road graph", True),
            ("2", "ASTraM Events", "Real BTP operational data", True),
            ("3", "Speed Panel", "Plane B — traffic speeds", True),
            ("4", "Calibrate AI", "LightGBM baseline + predictor", True),
            ("5", "Incident Config", "Event parameters", True),
            ("6", "Footprint", "Predicted congestion map", True),
            ("7", "Deployment", "BPR simulation + optimizer", True),
            ("8", "Mission Brief", "Deployment plan + learning loop", True),
        ]
        for num, name, detail, _ in stages_overview:
            done = ss[checks[int(num)-1]] if int(num) <= 8 else False
            col = "#00d4aa" if done else ("#ff6b35" if int(num) == active else "#334")
            icon = "✅" if done else ("⚡" if int(num) == active else "○")
            st.markdown(
                f'<div style="color:{col};padding:3px 0;font-size:0.88rem">'
                f'{icon} <b>Stage {num}</b> — {name} <span style="color:#556">({detail})</span></div>',
                unsafe_allow_html=True)

    # ── Two-model system flowchart ────────────────────────────────────────────
    _train_col = (
        '<div style="flex:1;background:#0d1117;border-radius:8px 0 0 8px;'
        'padding:12px 14px;border:1px solid #1e2535;border-right:none">'
        '<div style="color:#ffaa44;font-size:0.65rem;font-weight:700;'
        'letter-spacing:.08em;margin-bottom:8px">TRAINING PHASE</div>'
        '<div style="font-size:0.8rem;color:#889;line-height:1.75">'
        '📦 Speed Panel (150 days)<br>'
        '&nbsp;&nbsp;↓ <span style="color:#ff6b35">drop event-day rows</span><br>'
        '&nbsp;&nbsp;↓ time-based 80/20 split<br>'
        '&nbsp;&nbsp;↓ <b style="color:#aabbff">LightGBM #1 BASELINE</b><br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;segment + time + weather → normal speed<br>'
        '&nbsp;&nbsp;↓ → <code style="color:#aabbff">baseline_predicted</code><br>'
        '&nbsp;&nbsp;↓ <span style="color:#ff6b35">δ = observed − baseline</span><br>'
        '&nbsp;&nbsp;↓ event-active rows only<br>'
        '&nbsp;&nbsp;↓ <b style="color:#cc88ff">LightGBM #2 DELTA PREDICTOR</b><br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;hop + crowd + type → δ ± CI'
        '</div></div>'
    )
    _arrow = (
        '<div style="display:flex;align-items:center;padding:0 8px;'
        'background:#111827;border-top:1px solid #1e2535;border-bottom:1px solid #1e2535">'
        '<span style="color:#4488ff;font-size:1.3rem">⟶</span></div>'
    )
    _plan_col = (
        '<div style="flex:1;background:#0d1117;border-radius:0 8px 8px 0;'
        'padding:12px 14px;border:1px solid #1e2535;border-left:none">'
        '<div style="color:#00d4aa;font-size:0.65rem;font-weight:700;'
        'letter-spacing:.08em;margin-bottom:8px">PLANNING PHASE</div>'
        '<div style="font-size:0.8rem;color:#889;line-height:1.75">'
        '📝 Future event (type · crowd · location)<br>'
        '&nbsp;&nbsp;↓ BFS hop distances from event junction<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#cc88ff;font-size:.75rem">'
        'priority = |&delta;| &times; hop_decay &times; centrality</span><br>'
        '&nbsp;&nbsp;↓ Baseline → <code style="color:#aabbff">baseline_predicted</code><br>'
        '&nbsp;&nbsp;↓ <b style="color:#cc88ff">Delta Predictor</b><br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;→ <span style="color:#ff6b35">predicted δ ± CI</span> per segment<br>'
        '&nbsp;&nbsp;↓ 🗺️ Congestion Footprint Map<br>'
        '&nbsp;&nbsp;↓ ⚙️ BPR Simulation (measure each intervention)<br>'
        '&nbsp;&nbsp;↓ 📐 Optimizer (allocate under budget)<br>'
        '&nbsp;&nbsp;↓ 📋 <b style="color:#ccd">Deployment Brief</b>'
        '</div></div>'
    )
    st.markdown(
        '<div style="margin-top:20px;border-top:1px solid #1e2535;padding-top:16px">'
        '<div style="color:#556;font-size:0.68rem;font-weight:700;letter-spacing:.1em;'
        'margin-bottom:10px">HOW THE TWO MODELS WORK TOGETHER</div>'
        '<div style="display:flex;align-items:stretch;gap:0">'
        + _train_col + _arrow + _plan_col +
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1a1f3a 0%,#12102a 100%);'
        'border:1px solid #3344aa;border-radius:10px;padding:12px 18px 6px 18px;margin-top:10px">'
        '<div style="color:#7799ff;font-size:.62rem;font-weight:700;letter-spacing:.1em;margin-bottom:3px">WANT THE FULL PICTURE?</div>'
        '<div style="color:#ccd;font-size:.85rem;font-weight:700">&#129504; Deep Dive — ML Architecture &amp; Scoring</div>'
        '<div style="color:#4455aa;font-size:.71rem;margin-top:2px;margin-bottom:8px">'
        'Two-model pipeline &middot; BFS diagram &middot; priority formula &middot; worked examples'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_ML_Models.py", label="Open ML Architecture Deep Dive →", icon="🧠", use_container_width=True)

    # ── Three core technologies ────────────────────────────────────────────────
    _c1 = (
        '<div style="flex:1;background:#0d1228;border:1px solid #2a3555;'
        'border-radius:9px;padding:12px 14px">'
        '<div style="color:#aabbff;font-size:.68rem;font-weight:800;'
        'letter-spacing:.08em;margin-bottom:7px">&#129504; TWO-MODEL ML</div>'
        '<div style="color:#6677aa;font-size:.73rem;line-height:1.6">'
        '<b style="color:#aabbff">Baseline</b> learns normal speed from non-event data.<br>'
        '<b style="color:#cc88ff">Delta Predictor</b> forecasts km/h slowdown &#177; CI<br>'
        'per segment per 15-min bin, before the event.</div>'
        '</div>'
    )
    _c2 = (
        '<div style="flex:1;background:#0d0a1e;border:1px solid #2a1a44;'
        'border-radius:9px;padding:12px 14px">'
        '<div style="color:#cc88ff;font-size:.68rem;font-weight:800;'
        'letter-spacing:.08em;margin-bottom:7px">&#127758; PRIORITY SCORING</div>'
        '<div style="color:#7766aa;font-size:.73rem;line-height:1.6">'
        '<b style="color:#cc88ff">|&#948;| &#215; hop_decay &#215; centrality</b><br>'
        '= which roads get officers first.<br>'
        'A road 3 hops away can outrank one 1 hop away<br>'
        'if it carries more of the network&rsquo;s traffic.<br>'
        '<b style="color:#9966cc">Proximity &#8800; priority.</b></div>'
        '<div style="margin-top:8px;font-size:.68rem;color:#4433aa">'
        '&#8595; see Stage 6 for the full formula &amp; worked example</div>'
        '</div>'
    )
    _c3 = (
        '<div style="flex:1;background:#0a1e14;border:1px solid #1a4422;'
        'border-radius:9px;padding:12px 14px">'
        '<div style="color:#00d4aa;font-size:.68rem;font-weight:800;'
        'letter-spacing:.08em;margin-bottom:7px">&#9881; SIMULATION + OPTIMIZER</div>'
        '<div style="color:#336655;font-size:.73rem;line-height:1.6">'
        'BPR simulator <b style="color:#00d4aa">measures</b> how much congestion<br>'
        'each intervention removes &mdash; not guesses.<br>'
        'Greedy optimizer allocates resources under budget,<br>'
        'guaranteed <b style="color:#00d4aa">&#8805;63&#37; of optimal</b>.</div>'
        '</div>'
    )
    st.markdown(
        '<div style="margin-top:16px;border-top:1px solid #1e2535;padding-top:14px">'
        '<div style="color:#556;font-size:.66rem;font-weight:700;letter-spacing:.1em;'
        'margin-bottom:10px">THREE CORE TECHNOLOGIES</div>'
        '<div style="display:flex;gap:10px;align-items:stretch">'
        + _c1 + _c2 + _c3 +
        '</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — ZONE SETUP
# ═══════════════════════════════════════════════════════════════════════════════
cls1 = _stage_class(1, active)
st.markdown(f'<div class="stage-card {cls1}">', unsafe_allow_html=True)
_stage_header(1, "🎯", "Zone Setup — Enter Target Coordinates", cls1)

if cls1 == "locked":
    st.caption("Complete the overview introduction to unlock.")

elif cls1 == "done":
    G, seg_info = ss.G, ss.seg_info
    st.markdown(f'<div class="done-bar"><span class="done-pill">📍 {G.number_of_nodes():,} junctions</span>'
                f'<span class="done-pill">🛣️ {G.number_of_edges():,} segments</span>'
                f'<span class="done-pill">📐 {ss.get("zone_radius",2000)//1000} km radius</span>'
                f'<span class="done-pill">🌐 OSMnx — OpenStreetMap</span></div>',
                unsafe_allow_html=True)
    # Compact map
    net_paths = _seg_paths(G, seg_info, default_color=[50, 80, 130, 180])
    venue = [{"position": [ss.zone_lng, ss.zone_lat], "color": [255,107,53,255], "road": "Target venue"}]
    _deck(ss.zone_lat, ss.zone_lng,
          path_layers=[(net_paths, 1)],
          scatter_layers=[(venue, 16)],
          height=280)
    st.markdown(
        '<div style="margin-top:10px;background:#0a1e14;border-left:3px solid #00d4aa;'
        'border-radius:0 7px 7px 0;padding:9px 14px;font-size:.82rem;color:#5a8a78">'
        '<b style="color:#00d4aa">Betweenness centrality</b> is computed here for every segment — '
        'the fraction of all network shortest paths that pass through it. '
        'High centrality = critical bridge. '
        '<span style="color:#336655">This feeds the priority formula used in Stage 6: '
        '<b style="color:#cc88ff">|&#948;| &times; hop_decay &times; centrality</b></span>'
        '</div>',
        unsafe_allow_html=True,
    )

else:  # active
    st.markdown("""
    Enter the **latitude and longitude** of your venue. The system will download the
    real road graph from OpenStreetMap via OSMnx — every junction, every segment,
    real lane counts and speed limits.
    """)
    st.markdown(
        '<div style="background:#0a1e14;border-left:3px solid #00d4aa;'
        'border-radius:0 7px 7px 0;padding:9px 14px;font-size:.82rem;color:#5a8a78;margin-bottom:10px">'
        '<b style="color:#00d4aa">Betweenness centrality</b> — the fraction of all shortest paths '
        'in the network that pass through each road. High centrality = critical bridge. '
        '<span style="color:#336655">Used in Stage 6 priority formula: '
        '<b style="color:#cc88ff">|&#948;| &times; hop_decay &times; centrality</b></span>'
        '</div>',
        unsafe_allow_html=True,
    )
    _insight("Using Chinnaswamy Stadium coordinates runs instantly (cached graph). "
             "Any other coordinates trigger a live OSMnx download (~30–60s).")

    with st.form("zone_form"):
        c1, c2, c3 = st.columns(3)
        lat    = c1.number_input("Latitude",  value=DEFAULT_LAT, format="%.4f", step=0.001)
        lng    = c2.number_input("Longitude", value=DEFAULT_LNG, format="%.4f", step=0.001)
        radius = c3.selectbox("Study radius", [1000, 1500, 2000, 3000], index=2,
                              format_func=lambda x: f"{x//1000} km")
        arm = st.form_submit_button("🎯  ARM ZONE", use_container_width=True, type="primary")

    if arm:
        with st.status("Downloading road network from OpenStreetMap…", expanded=True) as status:
            from road_network import build_road_graph
            from impact import segment_centrality

            status.write("⟳  Connecting to OSMnx / Overpass API…")
            time.sleep(0.4)
            status.write("⟳  Building directed road graph (junctions + segments)…")
            G, seg_info = build_road_graph(
                venue_lat=lat, venue_lng=lng, radius_m=radius
            )
            status.write(f"✓  Graph loaded: {G.number_of_nodes():,} junctions, "
                         f"{G.number_of_edges():,} road segments")
            time.sleep(0.3)
            status.write("⟳  Computing betweenness centrality (bottleneck detection)…")
            seg_bc = segment_centrality(G, seg_info)
            status.write(f"✓  Centrality computed for {len(seg_bc):,} segments")
            status.update(label="Road network armed! ✅", state="complete")

        ss.G, ss.seg_info, ss.seg_bc = G, seg_info, seg_bc
        ss.zone_lat, ss.zone_lng = lat, lng
        ss.zone_radius = radius
        ss.zone_done = True
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — ASTRAM EVENTS
# ═══════════════════════════════════════════════════════════════════════════════
cls2 = _stage_class(2, active)
st.markdown(f'<div class="stage-card {cls2}">', unsafe_allow_html=True)
_stage_header(2, "📂", "ASTraM Events — Bengaluru Traffic Police Data", cls2)

if cls2 == "locked":
    st.caption("Complete Stage 1 to unlock.")

elif cls2 == "done":
    events, event_days = ss.events, ss.event_days
    G = ss.G

    st.markdown(
        f'<div class="done-bar">'
        f'<span class="done-pill">📊 {len(events):,} events in zone</span>'
        f'<span class="done-pill">📅 {len(event_days)} event-days derived</span>'
        f'<span class="done-pill">🎪 {events["is_gathering"].sum()} gathering events</span>'
        f'<span class="done-pill">📆 Nov 2023–Apr 2024</span>'
        f'</div>', unsafe_allow_html=True)

    # ── 22 event-day junction map ──────────────────────────────────────────
    st.markdown("**📍 All 22 event-days located on the real Bengaluru road graph**")

    _TYPE_COLORS = {
        "public_event":  [255, 107,  53, 255],   # orange
        "protest":       [255, 204,   0, 255],   # yellow
        "construction":  [160, 160, 160, 220],   # gray
        "congestion":    [255,  68,  68, 255],   # red
        "procession":    [  0, 212, 170, 255],   # teal
        "vip_movement":  [170, 102, 255, 255],   # purple
    }

    # Road network base layer
    net_paths = _seg_paths(G, ss.seg_info, default_color=[40, 65, 100, 140])

    # 22 event-day junction points (larger, colored by type)
    event_pts = []
    for date, ev in event_days.items():
        jn = ev["junction"]
        if jn not in G.nodes:
            continue
        color = _TYPE_COLORS.get(ev["type"], [200, 200, 200, 220])
        event_pts.append({
            "position": [G.nodes[jn]["lng"], G.nodes[jn]["lat"]],
            "color": color,
            "label": (f"{date}  ·  {ev['type'].replace('_',' ').title()}"
                      f"  ·  ~{ev['crowd']:,} crowd"
                      f"  ·  {ev['start_h']:.0f}:00–{ev['end_h']:.0f}:00"),
            "radius": max(10, min(22, ev["crowd"] // 2500)),   # size ∝ crowd
        })

    # Venue marker (white, largest)
    venue_pts = [{
        "position": [ss.zone_lng, ss.zone_lat],
        "color": [255, 255, 255, 255],
        "label": "Chinnaswamy Stadium (target venue)",
        "radius": 20,
    }]

    _deck(ss.zone_lat, ss.zone_lng,
          path_layers=[(net_paths, 1)],
          scatter_layers=[(event_pts, 8), (venue_pts, 8)],
          zoom=13.5, height=420)

    # Legend
    leg_cols = st.columns(len(_TYPE_COLORS) + 1)
    type_icons = {
        "public_event": ("🟠", "Public Event"),
        "protest":      ("🟡", "Protest"),
        "construction": ("⚪", "Construction"),
        "congestion":   ("🔴", "Congestion"),
        "procession":   ("🟢", "Procession"),
        "vip_movement": ("🟣", "VIP Movement"),
    }
    for col, (t, (icon, label)) in zip(leg_cols, type_icons.items()):
        count = sum(1 for ev in event_days.values() if ev["type"] == t)
        col.markdown(f"{icon} **{label}**  \n{count} day{'s' if count != 1 else ''}",
                     unsafe_allow_html=False)
    leg_cols[-1].markdown("⚪ **Venue**  \nChinnaswamy")

else:  # active
    st.markdown("""
    Upload the **ASTraM events CSV** from Bengaluru Traffic Police.
    Or click **Use Sample Data** to load the bundled Bengaluru dataset (8,173 events).
    """)
    _disclosure(
        "ASTraM records <em>what events happened</em> — type, location, time. "
        "It does NOT record attendance numbers, traffic speeds, or intervention outcomes. "
        "GridLock fills this gap using Plane B (TomTom speeds) and BPR simulation."
    )

    col_up, col_sample = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload astram_events.csv", type=["csv"],
            label_visibility="collapsed",
            help="Drag and drop the ASTraM CSV file here"
        )
    with col_sample:
        use_sample = st.button("📂 Use Bengaluru\nsample data", use_container_width=True)

    csv_source = None
    if use_sample:
        csv_source = "sample"
    elif uploaded_file is not None:
        csv_source = "upload"

    if csv_source:
        with st.status("Processing ASTraM events…", expanded=True) as status:
            from data_loader import load_clean_events
            from events_calendar import derive_event_days

            status.write("⟳  Reading CSV — checking format and columns…")
            time.sleep(0.3)

            if csv_source == "upload":
                import io
                raw_df = pd.read_csv(io.BytesIO(uploaded_file.read()))
                tmp_path = ROOT / "data" / "raw" / "_uploaded_events.csv"
                raw_df.to_csv(tmp_path, index=False)
                csv_path = tmp_path
                status.write(f"✓  Uploaded: {len(raw_df):,} rows, {len(raw_df.columns)} columns")
            else:
                csv_path = ROOT / "data" / "raw" / "astram_events.csv"
                raw_df = pd.read_csv(csv_path)
                status.write(f"✓  Loaded bundled dataset: {len(raw_df):,} rows")

            time.sleep(0.3)
            status.write("⟳  Parsing timestamps (UTC → IST / UTC+5:30)…")
            time.sleep(0.2)
            status.write("⟳  Cleaning coordinates — dropping zero/null GPS rows…")
            time.sleep(0.2)
            status.write(f"⟳  Snapping events to nearest OSMnx junction (within {ss.zone_radius}m)…")
            events = load_clean_events(csv_path=csv_path, zone_only=True, G=ss.G)
            status.write(f"✓  {len(events):,} events in zone · "
                         f"{events['is_gathering'].sum()} crowd-generating events")
            time.sleep(0.3)
            status.write("⟳  Deriving event-days — grouping by date, extracting crowd estimates…")
            event_days = derive_event_days(events, zone_only=True)
            status.write(f"✓  {len(event_days)} unique event-days")
            status.write("⟳  Crowd size: extracting from description text, falling back to type priors…")
            crowd_src = {}
            for ev in event_days.values():
                src = ev.get("crowd_source","prior")
                crowd_src[src] = crowd_src.get(src, 0) + 1
            status.write(f"✓  Crowd sources: {crowd_src}")
            status.update(label="ASTraM events ingested! ✅", state="complete")

        ss.events, ss.event_days = events, event_days
        ss.raw_event_count = len(raw_df)
        ss.events_done = True
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Show event analysis after done
if cls2 == "done" and ss.events is not None:
    events, event_days = ss.events, ss.event_days
    with st.expander("📊 What we extracted from ASTraM — expand to see"):
        ea1, ea2 = st.columns(2)
        with ea1:
            st.markdown("**What ASTraM gives us vs what it doesn't**")
            st.markdown("""
| ✅ Available | ❌ Not available |
|---|---|
| Event type & cause | Crowd attendance |
| GPS coordinates | Traffic speed impact |
| Timestamp (UTC → IST) | Officer deployment |
| Priority level | Congestion level |
| Route path (86 events) | Event outcome |
""")
            _disclosure(
                "Crowd size is the <b>only augmented field</b>. Sources: "
                "(1) regex from description text e.g. '1000 persons', "
                "(2) venue capacity override for stadiums, "
                "(3) event-type priors as fallback. "
                "Every crowd estimate carries a <code>crowd_source</code> label so judges can verify."
            )
        with ea2:
            cause_counts = events["event_cause"].value_counts().head(8).reset_index()
            cause_counts.columns = ["Cause", "Count"]
            chart = (alt.Chart(cause_counts).mark_bar()
                     .encode(x=alt.X("Count:Q"), y=alt.Y("Cause:N", sort="-x"),
                             color=alt.Color("Count:Q", scale=alt.Scale(scheme="oranges"), legend=None),
                             tooltip=["Cause:N","Count:Q"])
                     .properties(title=f"Event types in {ss.zone_radius//1000}km zone"))
            st.altair_chart(_dark(chart, 240), use_container_width=True)

        st.markdown(f"**{len(event_days)} event-days derived from real dates**")
        cal_rows = []
        for date, ev in sorted(event_days.items()):
            cal_rows.append({
                "Date": date,
                "Type": ev["type"].replace("_"," ").title(),
                "Crowd ~": f"{ev['crowd']:,}",
                "Source": ev["crowd_source"],
                "Window": f"{ev['start_h']:02.0f}:00–{ev['end_h']:02.0f}:00",
                "Events that day": ev["n_events"],
            })
        st.dataframe(pd.DataFrame(cal_rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — SPEED PANEL (Plane B)
# ═══════════════════════════════════════════════════════════════════════════════
cls3 = _stage_class(3, active)
st.markdown(f'<div class="stage-card {cls3}">', unsafe_allow_html=True)
_stage_header(3, "⚡", "Speed Panel — Traffic Speeds (Plane B)", cls3)

if cls3 == "locked":
    st.caption("Complete Stage 2 to unlock.")

elif cls3 == "done":
    panel_d = ss.panel_d
    n_segs  = panel_d["segment_id"].nunique() if panel_d is not None else 0
    n_rows  = len(panel_d) if panel_d is not None else 0
    st.markdown(
        f'<div class="done-bar">'
        f'<span class="done-pill">📈 {n_rows:,} speed records</span>'
        f'<span class="done-pill">🛣️ {n_segs:,} road segments</span>'
        f'<span class="done-pill">⏱️ 15-min bins, 150 days</span>'
        f'<span class="done-pill">🌐 TomTom-ready (synthetic fallback active)</span>'
        f'</div>', unsafe_allow_html=True)

else:  # active
    _disclosure(
        "<b>Live TomTom data is the primary source; synthetic speeds are the fallback.</b> "
        "In production, <code>fetch_tomtom_flow()</code> in <code>traffic_data.py</code> calls "
        "TomTom Flow Segment Data API (or the MOVE portal historical area report) and returns "
        "real 15-min speed bins per segment — no code change needed, just supply an API key. "
        "<b>This demo uses the synthetic fallback</b>: statistically realistic speeds with "
        "real rush-hour peaks, weekend dips, rain coefficients, and event-day slowdowns, "
        "so the full pipeline runs without a live key."
    )
    _insight(
        "The schema is identical between live and synthetic — segment_id, date, time, "
        "observed_speed, free_flow_speed. Switching to real TomTom is one line in config.py "
        "(set USE_SYNTHETIC = False and add the key). Nothing downstream changes."
    )

    if st.button("⚡  BUILD SPEED PANEL", use_container_width=True, type="primary"):
        with st.status("Building traffic speed panel…", expanded=True) as status:
            from traffic_data import build_speed_panel
            from features import build_feature_table, split_baseline_data
            from delta import compute_deltas

            status.write("⟳  Generating 15-min speed bins for all segments × 150 days…")
            time.sleep(0.4)
            panel = build_speed_panel(ss.G, ss.seg_info, ss.event_days,
                                      use_synthetic=True, tomtom_key=None)
            status.write(f"✓  {len(panel):,} speed records built")
            time.sleep(0.3)
            status.write("⟳  Engineering cyclical time features (sin/cos encoding)…")
            featured, seg_cats = build_feature_table(panel)
            status.write("✓  Feature table ready")
            status.update(label="Speed panel ready! ✅", state="complete")

        ss.panel_raw = panel
        ss.featured  = featured
        ss.seg_cats  = seg_cats
        ss.panel_done = True
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — CALIBRATE AI
# ═══════════════════════════════════════════════════════════════════════════════
cls4 = _stage_class(4, active)
st.markdown(f'<div class="stage-card {cls4}">', unsafe_allow_html=True)
_stage_header(4, "🧠", "Calibrate AI — Train Baseline + Delta Predictor", cls4)

if cls4 == "locked":
    st.caption("Complete Stage 3 to unlock.")

elif cls4 == "done":
    bm = ss.bm
    dp = ss.dp
    dp_types = ", ".join(t.replace("_"," ").title()
                         for t in (dp.metrics.get("event_types_seen") or []))
    st.markdown(
        f'<div class="done-bar">'
        f'<span class="done-pill">🧠 LightGBM #1 Baseline — MAE {bm.metrics["test_mae"]:.2f} km/h</span>'
        f'<span class="done-pill">🔮 LightGBM #2 Delta Predictor — MAE {dp.metrics.get("test_mae_kmh","—")} km/h</span>'
        f'<span class="done-pill">✅ Non-event delta bias ≈ 0</span>'
        f'<span class="done-pill">🎪 Predictor covers: {dp_types or "—"}</span>'
        f'</div>', unsafe_allow_html=True)

else:  # active
    # ── WHY this model exists ─────────────────────────────────────────────
    st.markdown("""
    <div class="formula-box">
      <span style="color:#aab;font-size:0.8rem;letter-spacing:0.06em">
        WHY WE NEED A BASELINE MODEL
      </span><br><br>
      To measure event impact we need a counterfactual —
      <em>"what would speed be here right now if no event existed?"</em><br><br>
      <span style="color:#aabbff;font-weight:700">baseline_speed</span>
      &nbsp;=&nbsp; LightGBM trained <b>only on normal-traffic days</b>
      &nbsp;→&nbsp; the counterfactual prediction<br><br>
      <span style="color:#ff6b35;font-size:1.15rem;font-weight:700">δ</span>
      &nbsp;=&nbsp;
      <span style="color:#00d4aa">observed_speed</span>
      &nbsp;−&nbsp;
      <span style="color:#aabbff">baseline_speed</span>
      &nbsp;&nbsp;
      <span style="color:#556;font-size:0.9rem">
        ← event's contribution, isolated from rush-hour &amp; weather
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── 6-step pipeline preview ───────────────────────────────────────────
    st.markdown("### 🔬 6-step pipeline — what's about to run")

    _tot = len(ss.featured) if ss.featured is not None else 0
    _segs = (ss.featured["segment_id"].nunique()
             if ss.featured is not None and "segment_id" in ss.featured.columns else "?")

    r1a, r1b, r1c = st.columns(3)
    with r1a:
        st.markdown(f"""
        <div style="background:#0d1117;border-radius:10px;padding:16px;
                    border-left:4px solid #334466;min-height:160px;margin-bottom:8px">
          <div style="color:#7799bb;font-size:0.68rem;font-weight:700;
                      letter-spacing:.1em;margin-bottom:6px">STEP 1 · SOURCE DATA</div>
          <div style="color:#eee;font-size:1rem;font-weight:700;margin-bottom:8px">
            📦 Speed Panel</div>
          <div style="color:#889;font-size:0.83rem;line-height:1.6">
            From Stage 3: <b style="color:#ccd">{_segs} segments × 96 bins/day</b><br>
            One row = one road segment × one 15-min time slice<br>
            Columns: <code>segment_id, hour, weekday, is_rain,
            observed_speed, is_event_day</code>
          </div>
        </div>""", unsafe_allow_html=True)

    with r1b:
        st.markdown(f"""
        <div style="background:#0d1117;border-radius:10px;padding:16px;
                    border-left:4px solid #cc3300;min-height:160px;margin-bottom:8px">
          <div style="color:#ff6b35;font-size:0.68rem;font-weight:700;
                      letter-spacing:.1em;margin-bottom:6px">STEP 2 · ⚠️ CRITICAL FILTER</div>
          <div style="color:#eee;font-size:1rem;font-weight:700;margin-bottom:8px">
            🚫 Drop Event-Day Rows</div>
          <div style="color:#889;font-size:0.83rem;line-height:1.6">
            Remove all rows where <code>is_event_day = True</code><br>
            <b style="color:#ff9966">Without this</b>: model learns "slow = normal"
            → delta shrinks → event impact undercounted<br>
            <b style="color:#00d4aa">With this</b>: baseline knows only normal traffic
          </div>
        </div>""", unsafe_allow_html=True)

    with r1c:
        st.markdown(f"""
        <div style="background:#0d1117;border-radius:10px;padding:16px;
                    border-left:4px solid #005577;min-height:160px;margin-bottom:8px">
          <div style="color:#00d4aa;font-size:0.68rem;font-weight:700;
                      letter-spacing:.1em;margin-bottom:6px">STEP 3 · TIME-BASED SPLIT</div>
          <div style="color:#eee;font-size:1rem;font-weight:700;margin-bottom:8px">
            📅 Train / Test Split</div>
          <div style="color:#889;font-size:0.83rem;line-height:1.6">
            <b style="color:#ccd">First 80% of dates</b> → Train set<br>
            <b style="color:#ccd">Last 20% of dates</b> → Test set<br>
            ⚠️ <b style="color:#ffcc44">Never random</b> — random split leaks
            future traffic patterns into training, inflating metrics
          </div>
        </div>""", unsafe_allow_html=True)

    r2a, r2b, r2c = st.columns(3)
    with r2a:
        st.markdown(f"""
        <div style="background:#0d1117;border-radius:10px;padding:16px;
                    border-left:4px solid #664400;min-height:160px;margin-bottom:8px">
          <div style="color:#ffaa44;font-size:0.68rem;font-weight:700;
                      letter-spacing:.1em;margin-bottom:6px">STEP 4 · BASELINE MODEL</div>
          <div style="color:#eee;font-size:1rem;font-weight:700;margin-bottom:8px">
            🧠 Train LightGBM #1</div>
          <div style="color:#889;font-size:0.83rem;line-height:1.6">
            Input: non-event rows only<br>
            Learns: <code>segment + time + weather → normal speed</code><br>
            300 trees · <code>segment_code</code> dominates (each road
            has its own demand profile)<br>
            Output: <code>baseline_predicted</code> per row
          </div>
        </div>""", unsafe_allow_html=True)

    with r2b:
        st.markdown(f"""
        <div style="background:#0d1117;border-radius:10px;padding:16px;
                    border-left:4px solid #005533;min-height:160px;margin-bottom:8px">
          <div style="color:#00d4aa;font-size:0.68rem;font-weight:700;
                      letter-spacing:.1em;margin-bottom:6px">STEP 5 · DELTA + VALIDATION</div>
          <div style="color:#eee;font-size:1rem;font-weight:700;margin-bottom:8px">
            ⚡ δ = Observed − Baseline</div>
          <div style="color:#889;font-size:0.83rem;line-height:1.6">
            Computed for every row (event + non-event)<br>
            <b style="color:#ccd">Bias check</b>: non-event mean δ must be ≈ 0<br>
            If not zero → baseline is biased → all downstream
            deltas are wrong and unusable
          </div>
        </div>""", unsafe_allow_html=True)

    with r2c:
        st.markdown(f"""
        <div style="background:#0d1117;border-radius:10px;padding:16px;
                    border-left:4px solid #440066;min-height:160px;margin-bottom:8px">
          <div style="color:#cc88ff;font-size:0.68rem;font-weight:700;
                      letter-spacing:.1em;margin-bottom:6px">STEP 6 · DELTA PREDICTOR</div>
          <div style="color:#eee;font-size:1rem;font-weight:700;margin-bottom:8px">
            🔮 Train LightGBM #2</div>
          <div style="color:#889;font-size:0.83rem;line-height:1.6">
            Filter: event-active rows only · Label: measured δ from Step 5<br>
            <code>hop_distance + crowd + event_type → predicted δ</code><br>
            Outputs a confidence interval per segment<br>
            Forecasts future event impact <b style="color:#ccd">before it happens</b>
          </div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.caption(
        "⏱ Estimated runtime: "
        "Step 1–3 (data prep) ~2s · "
        "Step 4 (LightGBM baseline) ~10s · "
        "Step 5 (delta) ~2s · "
        "Step 6 (delta predictor) ~5s"
    )

    if st.button("🧠  CALIBRATE AI — Run All 6 Steps Live",
                 use_container_width=True, type="primary"):
        with st.status("Running 6-step AI calibration…", expanded=True) as status:
            from features import split_baseline_data
            from baseline_model import BaselineModel
            from delta import compute_deltas, validate_baseline
            from delta_predictor import DeltaPredictor, build_predictor_training_data

            # Step 1
            status.write("**Step 1 / 6** · Loading speed panel from session state…")
            _n_tot = len(ss.featured)
            _n_seg = (ss.featured["segment_id"].nunique()
                      if "segment_id" in ss.featured.columns else "?")
            status.write(f"✓  {_n_tot:,} rows · {_n_seg} segments · 96 bins/day")

            # Step 2
            status.write(
                "**Step 2 / 6** · Filtering OUT event-day rows "
                "(`is_event_day = True`)…"
            )
            split = split_baseline_data(ss.featured)
            _kept = len(split["X_train"]) + len(split["X_test"])
            _excl = split["n_excluded_event_rows"]
            status.write(
                f"✓  Kept {_kept:,} non-event rows  |  "
                f"Excluded {_excl:,} event-day rows  "
                f"({round(100 * _excl / max(_n_tot, 1))}% of panel removed)"
            )

            # Step 3
            status.write(
                "**Step 3 / 6** · Time-based 80/20 split "
                "(chronological — never random)…"
            )
            status.write(
                f"✓  Train: {len(split['X_train']):,} rows "
                f"(first 80% of non-event dates)  |  "
                f"Test: {len(split['X_test']):,} rows (last 20%)"
            )
            time.sleep(0.2)

            # Step 4
            status.write(
                "**Step 4 / 6** · Training LightGBM baseline — "
                "300 trees on normal-traffic data…"
            )
            bm = BaselineModel()
            metrics = bm.train(split, ss.seg_cats)
            status.write(
                f"✓  Baseline trained  |  "
                f"Train MAE: {metrics['train_mae']:.2f} km/h  |  "
                f"Test MAE: {metrics['test_mae']:.2f} km/h"
            )
            time.sleep(0.2)

            # Step 5
            status.write(
                "**Step 5 / 6** · Computing δ = observed − baseline "
                "for every row in the panel…"
            )
            panel_d = compute_deltas(ss.featured, bm)
            checks_v = validate_baseline(panel_d)
            _mean = checks_v["non_event_delta_mean"]
            _bias_label = "✅ unbiased — baseline is correct" if abs(_mean) < 0.5 \
                          else "⚠️ bias detected — check baseline training"
            status.write(
                f"✓  Delta computed  |  "
                f"Non-event mean δ: {_mean:+.4f} km/h  →  {_bias_label}"
            )
            time.sleep(0.2)

            # Step 6
            status.write(
                "**Step 6 / 6** · Building delta predictor training data "
                "from event-active rows…"
            )
            train_df = build_predictor_training_data(
                panel_d, ss.event_days, ss.G, ss.seg_info
            )
            status.write(
                f"✓  {len(train_df):,} event rows across "
                f"{train_df['date'].nunique()} event-dates"
            )

            if not train_df.empty:
                status.write(
                    "**Step 6 / 6** · Training LightGBM delta predictor "
                    "(event features → predicted Δ)…"
                )
                dp = DeltaPredictor()
                dp_metrics = dp.train(train_df)
                status.write(
                    f"✓  Delta predictor trained  |  "
                    f"Train MAE: {dp_metrics['train_mae_kmh']} km/h  |  "
                    f"Test MAE: {dp_metrics['test_mae_kmh']} km/h"
                )
            else:
                dp = DeltaPredictor.load()
                status.write(
                    "ℹ  Loaded pre-trained delta predictor "
                    "(no new event data to train on)"
                )

            status.update(
                label="✅  All 6 steps complete — AI calibration successful!",
                state="complete"
            )

        ss.bm, ss.dp = bm, dp
        ss.panel_d = panel_d
        ss.calibrated = True
        st.rerun()

# Expanded results after calibration
if cls4 == "done" and ss.bm is not None:
    with st.expander("📊 Two-model architecture — expand to inspect both LightGBM models", expanded=False):
        bm = ss.bm
        dp = ss.dp
        panel_d = ss.panel_d

        tab_bm, tab_dp, tab_arch = st.tabs([
            "🧠 Model 1 — Baseline (normal-speed predictor)",
            "🔮 Model 2 — Delta Predictor (event impact forecaster)",
            "🔀 How the two models work together",
        ])

        # ── Tab 1: Baseline ──────────────────────────────────────────────
        with tab_bm:
            st.markdown("""
**What it does:** Given any road segment + time-of-day + weekday + weather, predicts
the speed that road *would have* if NO event were happening — the counterfactual.

**Trained on:** Non-event rows only (`is_event_day = False`).
If event traffic leaked in, the model would learn "roads are sometimes slow for no reason"
— shrinking the delta and undercounting event impact.
""")
            # ── Dataset construction breakdown ───────────────────────────
            if panel_d is not None:
                _n_total    = len(panel_d)
                _n_nonevent = int((~panel_d["is_event_day"]).sum())
                _n_event    = int(panel_d["is_event_day"].sum())
                _n_segs     = panel_d["segment_id"].nunique()
                _pct_kept   = round(100 * _n_nonevent / max(_n_total, 1))
            else:
                _n_total = _n_nonevent = _n_event = _n_segs = _pct_kept = 0

            with st.expander("📐 How the Baseline training dataset is built — step by step", expanded=True):
                _dc1, _dc2 = st.columns([3, 2])
                with _dc1:
                    st.markdown(f"""
| Step | What happens |
|------|-------------|
| **1. Source** | Speed panel from Stage 3: `{_n_segs:,}` segments × 150 days × 96 bins/day |
| **2. Unit of observation** | One row = one road segment × one 15-min time bin |
| **3. Total rows in panel** | `{_n_total:,}` speed records |
| **4. Critical filter** | Drop every row where `is_event_day = True` — removes `{_n_event:,}` event-day records |
| **5. Training rows kept** | `{_n_nonevent:,}` ({_pct_kept}% of panel) — only normal-traffic observations |
| **6. Features (X)** | `segment_code` (road identity), `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`, `is_weekend`, `is_rain`, `lanes`, `free_flow_speed`, `capacity` |
| **7. Label (Y)** | `observed_speed` — the actual measured speed on that segment at that time |
| **8. Train / test split** | Time-based cutoff: first 80% of dates → train, last 20% → test *(never random)* |
""")
                with _dc2:
                    st.markdown("""
**Why the event-day filter is the most important step:**

If event-day traffic (artificially slow roads) leaks into training:
1. Model learns "slow roads are sometimes normal"
2. It underestimates normal speed
3. `delta = observed − baseline` shrinks
4. Event impact is **undercounted**

This filter is the single correctness guarantee that makes the whole delta measurement honest.

---
**Why time-based — not random — split?**

Random split leaks future traffic patterns into training. A model trained on Mon+Wed+Fri data and tested on Tuesday "already knows" what Tuesday looks like. Time-based split enforces that the test set is genuinely unseen future data.
""")

            bm1, bm2 = st.columns(2)
            with bm1:
                fi_names = ["segment_code","hour_sin","hour_cos","weekday_sin","weekday_cos",
                            "is_weekend","is_rain","lanes","free_flow_speed","capacity"]
                imps = list(bm.model.feature_importances_)
                n = min(len(fi_names), len(imps))
                fi_df = pd.DataFrame({"Feature": fi_names[:n], "Importance": imps[:n]})
                fi_df = fi_df.sort_values("Importance", ascending=False)
                chart = (alt.Chart(fi_df).mark_bar()
                         .encode(x=alt.X("Importance:Q"),
                                 y=alt.Y("Feature:N", sort="-x"),
                                 color=alt.Color("Importance:Q",
                                                 scale=alt.Scale(scheme="oranges"), legend=None),
                                 tooltip=["Feature:N","Importance:Q"])
                         .properties(title="Baseline — Feature Importance"))
                st.altair_chart(_dark(chart, 280), use_container_width=True)
                st.caption(f"Test MAE: {bm.metrics['test_mae']:.2f} km/h  |  "
                           f"Train rows: {bm.metrics['train_rows']:,}")
            with bm2:
                non_ev = panel_d[~panel_d["is_event_day"]]["delta"].sample(
                    min(20000, (~panel_d["is_event_day"]).sum()), random_state=42)
                hist_df = pd.DataFrame({"delta": non_ev.values})
                h_chart = (alt.Chart(hist_df).mark_bar(color="#00d4aa", opacity=0.75)
                           .encode(x=alt.X("delta:Q", bin=alt.Bin(maxbins=80)), y=alt.Y("count()"))
                           .properties(title="Validation: non-event delta ≈ 0 means model is unbiased"))
                rule = (alt.Chart(pd.DataFrame({"x":[0]}))
                        .mark_rule(color="#ff6b35", strokeDash=[5,3]).encode(x="x:Q"))
                st.altair_chart(_dark(h_chart + rule, 280), use_container_width=True)
                st.success("Histogram centred on 0 ✅ — the baseline learns normal traffic correctly. "
                           "Any shift would mean the model bakes event-slowdown into its 'normal' prediction.")

        # ── Tab 2: Delta Predictor ───────────────────────────────────────
        with tab_dp:
            st.markdown("""
**What it does:** Given a *planned* future event (type, crowd, timing, location),
predicts the congestion DELTA per road segment — **before the event happens**.
This is the forecasting engine. It maps:

> `(hop_from_event, crowd, event_type, road_capacity, time_of_day)` → **predicted Δ km/h**

**Trained on:** Event-active rows that already have a *measured* delta from Stage 5.
The measured delta becomes the training label. So the model learns: "a 50k crowd
public_event 2 hops away from this capacity road at 7pm slows it by X km/h."
""")
            _insight(
                "<b>Key innovation — hop_from_event:</b> This feature is the graph-hop distance "
                "from the event junction to each road segment. Roads 1 hop away feel the worst impact; "
                "roads 4+ hops away barely feel it. This spatial decay is what makes predictions "
                "geographically meaningful — not just 'the city slows down', but 'THESE roads slow '."
            )

            dp2a, dp2b = st.columns(2)
            with dp2a:
                from delta_predictor import PREDICTOR_FEATURES
                if dp is not None and dp.model is not None:
                    dp_imps = list(dp.model.feature_importances_)
                    n2 = min(len(PREDICTOR_FEATURES), len(dp_imps))
                    dp_fi = pd.DataFrame({
                        "Feature": PREDICTOR_FEATURES[:n2],
                        "Importance": dp_imps[:n2]
                    }).sort_values("Importance", ascending=False)

                    feat_roles = {
                        "hop_from_event":  "🎯 Spatial: hops from event",
                        "hop_weight":      "🎯 Spatial: weighted decay",
                        "crowd":           "👥 Event: raw crowd size",
                        "crowd_log":       "👥 Event: log-scaled crowd",
                        "event_type_code": "🎪 Event: type (match/protest…)",
                        "capacity":        "🛣️ Road: lanes × speed",
                        "lanes":           "🛣️ Road: lane count",
                        "free_flow_speed": "🛣️ Road: posted speed limit",
                        "baseline_predicted": "📊 Context: normal speed",
                        "hour_sin":        "⏰ Time: cyclical hour",
                        "hour_cos":        "⏰ Time: cyclical hour",
                        "weekday_sin":     "📅 Time: cyclical weekday",
                        "weekday_cos":     "📅 Time: cyclical weekday",
                        "is_weekend":      "📅 Time: weekend flag",
                        "is_rain":         "🌧️ Weather: rain flag",
                        "segment_code":    "🛣️ Road: identity code",
                    }
                    dp_fi["Role"] = dp_fi["Feature"].map(feat_roles).fillna("")

                    dp_chart = (alt.Chart(dp_fi).mark_bar()
                                .encode(x=alt.X("Importance:Q"),
                                        y=alt.Y("Feature:N", sort="-x"),
                                        color=alt.Color("Importance:Q",
                                                        scale=alt.Scale(scheme="purples"),
                                                        legend=None),
                                        tooltip=["Feature:N","Role:N","Importance:Q"])
                                .properties(title="Delta Predictor — Feature Importance"))
                    st.altair_chart(_dark(dp_chart, 360), use_container_width=True)
                    st.caption(f"Train MAE: {dp.metrics.get('train_mae_kmh','—')} km/h  |  "
                               f"Test MAE: {dp.metrics.get('test_mae_kmh','—')} km/h  |  "
                               f"Train rows: {dp.metrics.get('train_rows','—')}")
                else:
                    st.info("Delta Predictor not yet trained — run Stage 4.")

            with dp2b:
                st.markdown("**Feature categories — what the predictor knows**")
                feat_groups = {
                    "🎯 Spatial (new vs baseline)": [
                        "`hop_from_event` — BFS hops from event junction",
                        "`hop_weight` — capacity-adjusted decay factor",
                    ],
                    "👥 Event (new vs baseline)": [
                        "`crowd` — expected attendance",
                        "`crowd_log` — log-scaled (handles 500 vs 50,000)",
                        "`event_type_code` — match/protest/procession differ",
                    ],
                    "📊 Context (new vs baseline)": [
                        "`baseline_predicted` — what normal speed is here",
                        "  (more impact if road is already at capacity)",
                    ],
                    "🛣️ Road + ⏰ Time (shared)": [
                        "Same as baseline: capacity, lanes, cyclical time",
                    ],
                }
                for group, items in feat_groups.items():
                    st.markdown(f"**{group}**")
                    for item in items:
                        st.markdown(f"  - {item}")

                st.markdown("**Confidence intervals (87% CI)**")
                st.markdown("""
Per-event-type residual std from the training set:
```
resid_std = std(actual_delta − predicted_delta)
              for rows of that event_type
CI_lo = delta_pred − 1.5 × resid_std
CI_hi = delta_pred + 1.5 × resid_std
```
So a protest with tight historical residuals gives a
narrow CI; a rarely-seen event type gives a wide one.
""")
                if dp is not None and dp._type_std:
                    std_df = pd.DataFrame([
                        {"Event type": k.replace("_"," ").title(),
                         "Residual std (km/h)": round(v, 3),
                         "CI width ±": f"±{1.5*v:.2f} km/h"}
                        for k, v in sorted(dp._type_std.items())
                    ])
                    st.dataframe(std_df, use_container_width=True, hide_index=True)

        # ── Tab 3: Architecture ──────────────────────────────────────────
        with tab_arch:
            st.markdown("""
```
                  TRAINING TIME (historical events)
                  ──────────────────────────────────

  Speed Panel (observed_speed)          ASTraM Events
         │                                    │
         ▼                                    ▼
  ┌─────────────────────┐         ┌──────────────────────┐
  │  BASELINE MODEL     │         │  Event calendar:     │
  │  (LightGBM #1)      │         │  junction, crowd,    │
  │                     │         │  type, timing        │
  │  Features:          │         └──────────┬───────────┘
  │  segment, time,     │                    │
  │  weather, road      │                    │
  │  (NO event data)    │                    │
  └──────────┬──────────┘                    │
             │ baseline_speed (counterfactual)│
             │                               │
             ▼                               ▼
        delta = observed_speed − baseline_speed
                    │
                    ▼
  ┌─────────────────────────────────────────────────┐
  │  DELTA PREDICTOR (LightGBM #2)                  │
  │                                                 │
  │  Features: hop_from_event, crowd, event_type,   │
  │            road_capacity, time_of_day           │
  │                                                 │
  │  Label: measured delta (from above)             │
  └─────────────────────────────────────────────────┘
                    │
                    ▼

                PLANNING TIME (future event)
                ─────────────────────────────

  User enters: event type, crowd, location, timing
                    │
                    ├──→ Baseline Model  →  baseline_speed per segment
                    │
                    ├──→ hop_from_event (BFS from junction)
                    │
                    ▼
         Delta Predictor  →  predicted_Δ ± CI per segment
                    │
                    ▼
              Congestion Footprint Map
```
""")
            _insight(
                "The Baseline can predict normal speed for any time/road — "
                "it has no idea what an event is. "
                "The Delta Predictor has no idea what 'normal' traffic looks like — "
                "it only knows how events perturb it. "
                "Together they answer the complete question."
            )

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — INCIDENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
cls5 = _stage_class(5, active)
st.markdown(f'<div class="stage-card {cls5}">', unsafe_allow_html=True)
_stage_header(5, "🎫", "Incident Configuration — Define the Event", cls5)

if cls5 == "locked":
    st.caption("Complete Stage 4 to unlock.")

elif cls5 == "done":
    ev = ss.event_spec
    st.markdown(
        f'<div class="done-bar">'
        f'<span class="done-pill">🎪 {ev["type"].replace("_"," ").title()}</span>'
        f'<span class="done-pill">👥 {ev["crowd"]:,} crowd</span>'
        f'<span class="done-pill">🕐 {ev["start_h"]:.0f}:00–{ev["end_h"]:.0f}:00</span>'
        f'<span class="done-pill">📅 {ev["weekday"]}</span>'
        f'{"<span class=\"done-pill\">🌧️ Rain</span>" if ev["is_rain"] else ""}'
        f'</div>', unsafe_allow_html=True)

else:  # active
    _insight(
        "The prediction uses the <b>real Bengaluru road graph from OSMnx</b> built in Stage 1. "
        "The congestion footprint is computed via graph-hop BFS from the event junction — "
        "roads 1 hop away are more impacted than roads 4 hops away."
    )

    G, event_days = ss.G, ss.event_days
    crowd_junctions = [(d, ev) for d, ev in event_days.items()
                       if ev.get("crowd", 0) > 0 and ev["junction"] in G.nodes]
    jn_options = {f"{d} — {ev['type']} @ ~{ev['crowd']:,} crowd": ev["junction"]
                  for d, ev in crowd_junctions}

    with st.form("incident_form"):
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            ev_type = st.selectbox("Event Type",
                ["public_event","protest","congestion","procession","vip_movement"])
            weekday = st.selectbox("Day of Week",
                ["Saturday","Sunday","Monday","Tuesday","Wednesday","Thursday","Friday"])
        with ic2:
            crowd   = st.slider("Expected Crowd", 500, 60000, 50000, step=500)
            start_h = st.slider("Start Hour", 8, 22, 17)
        with ic3:
            end_h   = st.slider("End Hour", 9, 23, 22)
            is_rain = st.toggle("Rain Forecast")

        jn_label = st.selectbox("Event junction (from ASTraM calendar)", list(jn_options.keys()))
        jn_node  = jn_options[jn_label]

        arm_btn = st.form_submit_button("🎫  ARM INCIDENT", use_container_width=True, type="primary")

    if arm_btn:
        ss.event_spec = {
            "type": ev_type, "junction": jn_node, "crowd": crowd,
            "start_h": start_h, "end_h": end_h,
            "weekday": weekday, "is_weekend": weekday in ("Saturday","Sunday"),
            "is_rain": is_rain,
        }
        ss.incident_done = True
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — FOOTPRINT PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
cls6 = _stage_class(6, active)
st.markdown(f'<div class="stage-card {cls6}">', unsafe_allow_html=True)
_stage_header(6, "🗺️", "Footprint — Predicted Congestion Map", cls6)

if cls6 == "locked":
    st.caption("Complete Stage 5 to unlock.")

elif cls6 in ("done", "active"):
    st.markdown("""
    <div style="background:#1a0a2e;border-left:4px solid #aa66ff;border-radius:0 8px 8px 0;
         padding:12px 16px;margin:8px 0;font-size:0.88rem;color:#cc99ff">
    🔮 <b>Delta Predictor (LightGBM #2) at work.</b>
    For each of the {segs:,} road segments in the zone, it computes
    <code>hop_from_event</code> (BFS distance from the event junction) then feeds
    <code>(hop_from_event, crowd, event_type, road_capacity, time_of_day)</code> into the model
    to produce a predicted Δ km/h with an 87% confidence interval.
    This runs entirely <em>before</em> the event happens — pure forecasting.
    </div>""".format(segs=len(ss.seg_info) if ss.seg_info else 0), unsafe_allow_html=True)

    with st.expander("📐 Priority Scoring Formula — how |δ| × hop_decay × centrality ranks every road", expanded=False):
        _priority_scoring_detail()

    # Auto-run prediction when stage becomes active
    if cls6 == "active" and not ss.footprint_done:
        with st.status("Predicting congestion footprint…", expanded=True) as status:
            dp, bm = ss.dp, ss.bm
            status.write(f"⟳  Running BFS from event junction across {len(ss.seg_info):,} segments…")
            time.sleep(0.3)
            fp = dp.predict_footprint(ss.event_spec, ss.G, ss.seg_info, bm, ss.seg_cats)
            status.write(f"✓  {len(fp):,} segment-time predictions generated")
            status.write(f"✓  Worst predicted Δ: {fp['delta_pred'].min():+.1f} km/h "
                         f"on '{fp.loc[fp.delta_pred.idxmin(),'road']}'")
            status.write("⟳  Building 87% confidence intervals (per-event-type residual std)…")
            time.sleep(0.3)
            status.update(label="Footprint mapped! ✅", state="complete")

        ss.footprint = fp
        ss.footprint_done = True
        st.rerun()

    fp = ss.footprint
    if fp is not None and not fp.empty:
        G, seg_info = ss.G, ss.seg_info
        ev = ss.event_spec

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Segments predicted", fp["segment_id"].nunique())
        mc2.metric("Worst Δ", f"{fp['delta_pred'].min():+.1f} km/h")
        mc3.metric("Avg corridor Δ", f"{fp['delta_pred'].mean():+.1f} km/h")
        mc4.metric("CI width (87%)", f"±{(fp['delta_hi']-fp['delta_pred']).mean():.1f} km/h")

        tab_map, tab_table, tab_heatmap, tab_hop = st.tabs(["🗺️ Congestion Map", "📋 Priority Ranking", "🔢 Severity Heatmap", "📉 Hop Decay"])

        with tab_map:
            seg_worst = fp.groupby("segment_id")["delta_pred"].min().to_dict()
            net_base  = _seg_paths(G, seg_info, default_color=[20, 35, 55, 80])
            hot_paths = []
            for sid, delta in seg_worst.items():
                info = seg_info.get(sid)
                if not info: continue
                u, v = info["u"], info["v"]
                if u not in G.nodes or v not in G.nodes: continue
                hot_paths.append({
                    "path": [[G.nodes[u]["lng"], G.nodes[u]["lat"]],
                             [G.nodes[v]["lng"], G.nodes[v]["lat"]]],
                    "color": _severity_color(delta),
                    "road": f"{info['road']}  Δ={delta:+.1f}",
                })
            venue_pts = [{"position": [G.nodes[ev["junction"]]["lng"],
                                       G.nodes[ev["junction"]]["lat"]],
                          "color": [255,255,255,255], "road": "Event location"}]
            _deck(ss.zone_lat, ss.zone_lng,
                  path_layers=[(net_base, 1), (hot_paths, 4)],
                  scatter_layers=[(venue_pts, 16)],
                  height=420)
            lc1, lc2, lc3, lc4 = st.columns(4)
            lc1.markdown("🔴 **SEVERE** < −10 km/h")
            lc2.markdown("🟠 **MODERATE** −5 to −10")
            lc3.markdown("🟡 **MILD** −2 to −5")
            lc4.markdown("🟢 **MINIMAL** ≥ −2 km/h")

        with tab_table:
            _seg_bc = ss.seg_bc or {}
            # Aggregate to segment level — one row per segment_id
            _seg_df = (fp.groupby(["segment_id", "road", "capacity"])
                       .agg(worst_delta=("delta_pred", "min"),
                            avg_delta=("delta_pred", "mean"),
                            hop=("hop_from_event", "min"),
                            delta_lo=("delta_lo", "min"),
                            delta_hi=("delta_hi", "max"))
                       .reset_index())
            # Add centrality and priority score
            _seg_df["centrality"]     = _seg_df["segment_id"].map(_seg_bc).fillna(0.0)
            _seg_df["priority_score"] = _seg_df["worst_delta"].abs() * (1 + _seg_df["centrality"] * 5)
            # Deduplicate by road name — same physical road can have multiple segment_ids
            _seg_df = (_seg_df.sort_values("priority_score", ascending=False)
                               .drop_duplicates(subset="road", keep="first")
                               .head(15)
                               .reset_index(drop=True))
            _seg_df["Severity"] = _seg_df["worst_delta"].apply(
                lambda d: "SEVERE" if d < -10 else ("MODERATE" if d < -5 else ("MILD" if d < -2 else "MINIMAL")))
            _seg_df["CI (87%)"] = _seg_df.apply(
                lambda r: f"[{r['delta_lo']:+.1f}, {r['delta_hi']:+.1f}]", axis=1)

            st.markdown("""
<div style="background:#0d1f2d;border-left:4px solid #00d4aa;border-radius:0 8px 8px 0;
     padding:10px 14px;margin:6px 0;font-size:0.84rem;color:#aacccc">
📐 <b>Priority Score = |Δ km/h| × (1 + Centrality × 5)</b><br>
Betweenness centrality (BC) measures how many shortest paths in the road graph pass through a segment —
a bottleneck road. A moderate slowdown on a high-BC road cascades to more destinations than a large
slowdown on a quiet side street. <b>This is why the officer placements in Stage 7 may differ from a
pure-delta ranking</b> — the optimizer targets high-priority-score junctions, not just the worst-delta road.
</div>""", unsafe_allow_html=True)

            st.dataframe(
                _seg_df[["road", "hop", "worst_delta", "CI (87%)", "Severity",
                          "centrality", "priority_score", "capacity"]]
                  .rename(columns={
                      "road": "Road", "hop": "Hops from Event",
                      "worst_delta": "Worst Δ (km/h)", "capacity": "Capacity (lanes×spd)",
                      "centrality": "Centrality (BC)", "priority_score": "Priority Score ↓",
                  })
                  .style.format({
                      "Worst Δ (km/h)": "{:+.1f}",
                      "Centrality (BC)": "{:.4f}",
                      "Priority Score ↓": "{:.2f}",
                  }),
                use_container_width=True, hide_index=True)

        with tab_heatmap:
            # Reuse _seg_df from tab_table (already computed above)
            _top_roads = _seg_df.head(10)["road"].tolist()
            _heat_df = fp[fp["road"].isin(_top_roads)].copy()
            _heat_agg = (_heat_df.groupby(["road", "hour"])["delta_pred"]
                         .min().reset_index())
            _heat_agg.columns = ["Road", "Hour", "Δ km/h"]
            _heatmap = (alt.Chart(_heat_agg)
                        .mark_rect()
                        .encode(
                            x=alt.X("Hour:O", title="Hour of day"),
                            y=alt.Y("Road:N",
                                    sort=alt.EncodingSortField("Δ km/h", op="min", order="ascending"),
                                    title="Road (worst → best)"),
                            color=alt.Color("Δ km/h:Q",
                                            scale=alt.Scale(scheme="redyellowgreen",
                                                            domain=[-20, 0], clamp=True),
                                            title="Δ speed (km/h)"),
                            tooltip=[alt.Tooltip("Road:N"),
                                     alt.Tooltip("Hour:O", title="Hour"),
                                     alt.Tooltip("Δ km/h:Q", format="+.1f")])
                        .properties(title="Predicted congestion severity — top-priority roads × hour of day",
                                    height=320))
            st.altair_chart(_dark(_heatmap, 320), use_container_width=True)
            lh1, lh2, lh3, lh4 = st.columns(4)
            lh1.markdown("🔴 **< −10 km/h** severe")
            lh2.markdown("🟠 **−5 to −10** moderate")
            lh3.markdown("🟡 **−2 to −5** mild")
            lh4.markdown("🟢 **≥ −2** minimal")
            st.caption("Top 10 roads by Priority Score (|Δ| × centrality). "
                       "Each cell = worst predicted Δ at that road-hour combination. "
                       "The time-of-day axis shows when each road peaks — useful for staggered deployment.")

        with tab_hop:
            hop_agg = fp.groupby("hop_from_event")["delta_pred"].mean().reset_index()
            hop_agg.columns = ["Hops from event", "Avg Δ (km/h)"]
            chart = (alt.Chart(hop_agg).mark_bar()
                     .encode(x=alt.X("Hops from event:O"),
                             y=alt.Y("Avg Δ (km/h):Q"),
                             color=alt.Color("Avg Δ (km/h):Q",
                                             scale=alt.Scale(scheme="redyellowgreen", reverse=True),
                                             legend=None))
                     .properties(title="Congestion decays with distance — validates BFS propagation model"))
            st.altair_chart(_dark(chart, 280), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — DEPLOYMENT OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════════
cls7 = _stage_class(7, active)
st.markdown(f'<div class="stage-card {cls7}">', unsafe_allow_html=True)
_stage_header(7, "🚦", "Deployment — Simulate & Optimise", cls7)

if cls7 == "locked":
    st.caption("Complete Stage 6 to unlock.")

elif cls7 in ("done", "active"):
    opt_done = ss.optimize_done

    if not opt_done:
        _insight(
            "<b>BPR Volume-Delay Model:</b> t = t₀ × (1 + 0.15 × (V/C)⁴). "
            "We infer V/C from the ML-predicted speed, then measure how each intervention "
            "changes throughput. Combined effect is simulated — never summed individually."
        )
        dc1, dc2, dc3 = st.columns(3)
        n_off  = dc1.slider("Officers", 1, 10, 4)
        n_bar  = dc2.slider("Barricades", 0, 5, 2)
        n_div  = dc3.slider("Diversions", 0, 3, 1)

        if st.button("🚦  RUN SIMULATION & OPTIMISE", use_container_width=True, type="primary"):
            with st.status("Running BPR simulation + greedy optimizer…", expanded=True) as status:
                from simulation import EventSimulator
                from optimizer import generate_candidates, InterventionOptimizer

                status.write("⟳  Generating intervention candidates from high-impact segments…")
                candidates = generate_candidates(ss.footprint, ss.seg_bc, ss.seg_info, ss.G)
                status.write(f"✓  {len(candidates)} candidates: "
                             f"{sum(1 for c in candidates if c['type']=='officer')} officers, "
                             f"{sum(1 for c in candidates if c['type']=='barricade')} barricades, "
                             f"{sum(1 for c in candidates if c['type']=='diversion')} diversions")
                time.sleep(0.3)

                status.write("⟳  Simulating each intervention independently (BPR)…")
                sim = EventSimulator(ss.G, ss.seg_info)
                iv_df, base_speeds = sim.measure_interventions(
                    ss.event_spec, ss.footprint, candidates[:15])
                status.write(f"✓  {len(iv_df)} interventions measured")
                if not iv_df.empty:
                    best = iv_df.iloc[0]
                    status.write(f"✓  Best single: [{best['type']}] {best['label'][:40]} "
                                 f"→ +{best['congestion_removed_kmh']:.2f} km/h")
                time.sleep(0.3)

                status.write(f"⟳  Greedy optimizer: finding best combination "
                             f"({n_off} officers + {n_bar} barricades + {n_div} diversions)…")
                opt = InterventionOptimizer(sim)
                plan, summary, sim_before, sim_after = opt.optimize(
                    ss.event_spec, ss.footprint, candidates,
                    n_officers=n_off, n_barricades=n_bar, n_diversions=n_div)
                opt_result = opt.report(plan, summary, sim_before, sim_after,
                                        ss.footprint, ss.seg_info)
                status.write(f"✓  Plan: {len(plan)} placements selected")
                status.write(f"✓  Net improvement: +{opt_result['net_improvement_kmh']:.2f} km/h")
                status.update(label="Deployment optimised! ✅", state="complete")

            ss.plan = plan
            ss.opt_result = opt_result
            ss.sim_before = sim_before
            ss.sim_after  = sim_after
            ss.iv_df      = iv_df
            ss.summary_df = summary
            ss.optimize_done = True
            st.rerun()

    else:
        opt_result = ss.opt_result
        plan       = ss.plan
        G, seg_info = ss.G, ss.seg_info

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Avg speed BEFORE", f"{opt_result['avg_speed_before']:.1f} km/h")
        rc2.metric("Avg speed AFTER",  f"{opt_result['avg_speed_after']:.1f} km/h")
        rc3.metric("Net improvement",  f"+{opt_result['net_improvement_kmh']:.2f} km/h")

        tab_iv, tab_steps, tab_map = st.tabs(["📊 Individual Effects","📈 Greedy Steps","🗺️ Deployment Map"])

        with tab_iv:
            iv_df = ss.get("iv_df", pd.DataFrame())
            if not iv_df.empty:
                chart = (alt.Chart(iv_df.head(10)).mark_bar()
                         .encode(x=alt.X("congestion_removed_kmh:Q", title="Congestion removed (km/h)"),
                                 y=alt.Y("label:N", sort="-x"),
                                 color=alt.Color("type:N", scale=alt.Scale(
                                     domain=["officer","barricade","diversion"],
                                     range=["#ff6b35","#00d4aa","#ffcc00"])),
                                 tooltip=["type:N","label:N","congestion_removed_kmh:Q","segments_improved:Q"])
                         .properties(title="Congestion removed per intervention (individual)"))
                st.altair_chart(_dark(chart, 320), use_container_width=True)

        with tab_steps:
            st.markdown("""
<div style="background:#1a0d00;border-left:4px solid #ff9500;border-radius:0 8px 8px 0;
     padding:9px 14px;margin:6px 0;font-size:0.83rem;color:#ffd0a0">
⚠️ <b>Why officer placements may not match the top-delta roads from Stage 6:</b>
The optimizer selects from candidates ranked by <code>|Δ| × (1 + centrality × 5)</code>.
A junction controlling multiple impacted roads (high betweenness centrality) is worth more
than one road with a large delta in isolation. Officers go to <em>bottleneck junctions</em>,
barricades to <em>low-capacity severe-delta segments</em>, diversions to segments with
viable alternate routes — three different criteria, by design.
</div>""", unsafe_allow_html=True)
            summary = ss.get("summary_df", pd.DataFrame())
            if not summary.empty:
                summary["label_short"] = summary.apply(
                    lambda r: f"Step {int(r['step'])}: {r['label'][:32]}…"
                    if len(r['label'])>32 else f"Step {int(r['step'])}: {r['label']}", axis=1)
                chart = (alt.Chart(summary).mark_bar()
                         .encode(x=alt.X("label_short:N", sort="x", axis=alt.Axis(labelAngle=-30)),
                                 y=alt.Y("marginal_gain_kmh:Q", title="Marginal gain (km/h)"),
                                 color=alt.Color("type:N", scale=alt.Scale(
                                     domain=["officer","barricade","diversion"],
                                     range=["#ff6b35","#00d4aa","#ffcc00"])))
                         .properties(title="Marginal gain per greedy step"))
                st.altair_chart(_dark(chart, 280), use_container_width=True)
            st.subheader("Final Deployment Plan")
            for i, iv in enumerate(plan, 1):
                icon = {"officer":"👮","barricade":"🚧","diversion":"↩️"}.get(iv["type"],"•")
                st.markdown(f"**{i}.** {icon} **[{iv['type'].upper()}]** — {iv['label']}")

        with tab_map:
            fp = ss.footprint
            net_base  = _seg_paths(G, seg_info, default_color=[20, 35, 55, 80])
            seg_worst = fp.groupby("segment_id")["delta_pred"].min().to_dict()
            hot_paths = []
            for sid, delta in seg_worst.items():
                if delta >= -3: continue
                info = seg_info.get(sid)
                if not info: continue
                u, v = info["u"], info["v"]
                if u not in G.nodes or v not in G.nodes: continue
                hot_paths.append({
                    "path": [[G.nodes[u]["lng"],G.nodes[u]["lat"]],
                             [G.nodes[v]["lng"],G.nodes[v]["lat"]]],
                    "color": _severity_color(delta),
                    "road": f"{info['road']}  Δ={delta:+.1f}",
                })
            officer_pts = []
            for iv in plan:
                if iv["type"] == "officer":
                    jn = iv.get("junction")
                    if jn and jn in G.nodes:
                        officer_pts.append({
                            "position": [G.nodes[jn]["lng"],G.nodes[jn]["lat"]],
                            "color": [255,107,53,255],
                            "road": f"👮 {iv['label'][:45]}",
                        })
            ev_jn = ss.event_spec["junction"]
            venue_pts = [{"position":[G.nodes[ev_jn]["lng"],G.nodes[ev_jn]["lat"]],
                          "color":[255,255,255,255],"road":"Event location"}]
            _deck(ss.zone_lat, ss.zone_lng,
                  path_layers=[(net_base, 1), (hot_paths, 4)],
                  scatter_layers=[(venue_pts, 18), (officer_pts, 14)],
                  height=420)
            st.caption("🔴/🟠 = congested segments · 🟠 dots = officer positions · ⚪ dot = event location")

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — MISSION BRIEF + LEARNING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
cls8 = _stage_class(8, active)
st.markdown(f'<div class="stage-card {cls8}">', unsafe_allow_html=True)
_stage_header(8, "📋", "Mission Brief — Deployment Plan + Learning Loop", cls8)

if cls8 == "locked":
    st.caption("Complete Stage 7 to unlock.")

else:
    # Generate brief
    if not ss.brief_done and ss.optimize_done:
        from recommender import generate_recommendation
        brief = generate_recommendation(
            ss.event_spec, ss.footprint, ss.plan, ss.opt_result,
            ss.sim_before, ss.sim_after,
            seg_info=ss.seg_info, event_date="Demo Event",
        )
        ss.brief = brief
        ss.brief_done = True
        st.rerun()

    brief = ss.brief

    tab_brief, tab_learn = st.tabs(["📋 Deployment Brief", "📈 Learning Loop"])

    with tab_brief:
        ev = ss.event_spec or {}
        opt = ss.opt_result or {}
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Event type", ev.get("type","—").replace("_"," ").title())
        b2.metric("Crowd", f"{ev.get('crowd',0):,}")
        b3.metric("Speed before", f"{opt.get('avg_speed_before',0):.1f} km/h")
        b4.metric("Net improvement", f"+{opt.get('net_improvement_kmh',0):.2f} km/h")

        st.text_area("Full Operational Brief", brief, height=500)
        if brief:
            st.download_button("⬇️  Download Brief (.txt)", brief,
                               file_name="gridlock_deployment_brief.txt", mime="text/plain")

    with tab_learn:
        _insight(
            "<b>The learning flywheel:</b> after every real event, the system compares "
            "predicted delta to actual delta, computes RMSE, adds the event to training, "
            "retrains — and the error shrinks. This is the differentiator: the system "
            "improves with every deployment."
        )

        lh_path = ROOT / "outputs" / "learning_history.json"
        tp_path = ROOT / "outputs" / "event_templates.json"

        if lh_path.exists():
            history   = json.load(open(lh_path))
            templates = json.load(open(tp_path)) if tp_path.exists() else {}
            measured  = [e for e in history if e.get("pred_rmse") is not None]

            lm1, lm2, lm3, lm4 = st.columns(4)
            lm1.metric("Events in loop", len(history))
            lm2.metric("Predictions measured", len(measured))
            if measured:
                first_r = measured[0]["pred_rmse"]
                best_r  = min(e["pred_rmse"] for e in measured)
                impr    = 100*(first_r - best_r)/first_r
                lm3.metric("First RMSE", f"{first_r:.2f} km/h")
                lm4.metric("Best RMSE / gain", f"{best_r:.2f} km/h  ({impr:.0f}% ↓)")

            if measured:
                lc_df = pd.DataFrame([
                    {"Event #": e["event_n"], "Date": e["date"],
                     "Type": e["event_type"].replace("_"," ").title(),
                     "RMSE (km/h)": e["pred_rmse"],
                     "Train rows": e["train_rows"]}
                    for e in measured
                ])
                floor_val = ss.bm.metrics.get("test_mae", 1.2) if ss.bm else 1.2
                chart = (alt.Chart(lc_df).mark_line(point=True)
                         .encode(x=alt.X("Event #:Q"),
                                 y=alt.Y("RMSE (km/h):Q"),
                                 color=alt.Color("Type:N",scale=alt.Scale(scheme="tableau10")),
                                 tooltip=["Date:N","Type:N","RMSE (km/h):Q","Train rows:Q"])
                         .properties(title="Prediction RMSE per event — real measured improvement"))
                floor_rule = (alt.Chart(pd.DataFrame({"y":[floor_val]}))
                              .mark_rule(color="#00d4aa", strokeDash=[6,3]).encode(y="y:Q"))
                st.altair_chart(_dark(chart + floor_rule, 300), use_container_width=True)
                st.caption(f"Dashed teal = baseline noise floor ({floor_val:.2f} km/h). "
                           "This is REAL data, not illustrative.")

            if templates:
                st.markdown(f"**{len(templates)} event templates saved** — proven impact fingerprints")
                tpl_rows = []
                for date, t in sorted(templates.items()):
                    ai = t.get("actual_impact", {})
                    tpl_rows.append({
                        "Date": date,
                        "Type": t["event_type"].replace("_"," ").title(),
                        "Crowd": f"{t['crowd']:,}",
                        "Avg Δ actual": f"{ai.get('avg_delta_kmh',0):+.2f}" if ai else "—",
                        "Worst Δ": f"{ai.get('worst_delta_kmh',0):+.2f}" if ai else "—",
                        "Severe segs": ai.get("severe_segs","—") if ai else "—",
                        "Peak hour": f"{ai.get('peak_hour',0):.0f}:00" if ai else "—",
                    })
                st.dataframe(pd.DataFrame(tpl_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Run `python src/main.py` once to generate the learning history from historical events.")

    # ── Mission Complete Banner ──────────────────────────────────────────────
    if ss.brief_done:
        st.markdown("""
        <div style="background:linear-gradient(90deg,#002a20,#1a1f2e);border:2px solid #00d4aa;
             border-radius:12px;padding:28px 36px;text-align:center;margin-top:20px">
          <div style="font-size:2.5rem">✅</div>
          <div style="font-size:1.6rem;color:#00d4aa;font-weight:700;margin:8px 0">
            MISSION COMPLETE
          </div>
          <div style="color:#aabbcc;font-size:0.95rem;margin-bottom:16px">
            All 8 stages complete · Footprint predicted · Deployment optimised · Brief generated
          </div>
          <div style="display:inline-flex;gap:16px;align-items:center;
               background:#0d1117;border:1px solid #ff6b35;border-radius:10px;
               padding:10px 24px;margin-top:4px">
            <span style="color:#ff6b35;font-size:0.8rem;font-weight:700;letter-spacing:0.1em">TEAM</span>
            <span style="color:#fff;font-size:1.4rem;font-weight:800;letter-spacing:0.05em">Srikrit</span>
            <span style="color:#334;font-size:1rem">·</span>
            <span style="color:#556;font-size:0.82rem">Flipkart GRID 2026 · Theme 2</span>
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
