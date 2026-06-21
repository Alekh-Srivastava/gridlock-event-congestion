# CLAUDE.md — Project Handoff & Context

> **Read this first.** This file tells you (Claude Code) everything about this
> project: the goal, the architecture, the current state, what is real vs.
> placeholder, the corrections to make, and what to build next. Keep this file
> updated as the project evolves.

---

## 1. What this project is (the one-paragraph version)

This is a submission for **Flipkart GRID 2026, Theme 2: Event-Driven Congestion**.
The goal is an **AI-powered traffic decision-intelligence system** for Bengaluru
that, given a planned or unplanned event (cricket match, procession, rally,
construction), **predicts the traffic congestion it will cause before it happens**,
shows exactly **which roads and time-windows** are hit, and **recommends + proves**
the optimal deployment of police officers, barricades, and diversions. It must go
beyond "predict congestion" to answer *"what will happen, why, and what should we
do about it."*

The judging context: ~1600 teams in Round 2, only a few reach the finals at
Flipkart HQ. The solution must be technically deep, **honest** (no fabricated
targets), and demonstrate real operational impact. Judges may know the data source
(it's Bengaluru Traffic Police's ASTraM platform), so we must not reinvent what
ASTraM already does (it already has microsimulation). Our differentiator is the
**event-specific decision layer**: quantifying impact in advance, optimizing
deployment, and learning after each event — the three gaps the problem statement
explicitly names.

---

## 2. The core mental model (DO NOT VIOLATE THIS)

The single most important design principle, which keeps the project honest:

```
ML predicts the PROBLEM.        Simulation proves the SOLUTION.
(how much congestion an event   (how much congestion each
 will cause)                     intervention removes)
```

**Congestion is never invented as a target.** It is measured as a DELTA:

```
delta = observed_speed  −  baseline_speed
        (REAL, measured)    (model's "no-event" counterfactual prediction)
```

- `observed_speed` = the actual measured speed on the road during the event
  (from a traffic API — Plane B). **Always real, never predicted.**
- `baseline_speed` = what the model predicts the speed WOULD have been at that
  place/time/weather IF NO EVENT happened (the counterfactual).
- `delta` < 0 means the event slowed traffic. **That negative number is the
  event's quantified impact** — the thing nobody produces today.

Why this matters: a judge will ask "how do you know the event caused this and not
just rush hour / rain?" The answer must always be: *"the baseline already accounts
for rush hour, weekday, and rain — it's trained only on non-event data — so the
delta is purely the event's contribution."*

---

## 3. The three data planes

| Plane | What it is | Source | Status in repo |
|-------|-----------|--------|----------------|
| **A — Events** | What disrupts traffic (type, location, time, cause) | ASTraM CSV | ✅ **REAL** — `data/raw/astram_events.csv` (8,173 rows, Bengaluru, Nov 2023–Apr 2024) |
| **B — Traffic** | Speed per road segment per 15-min bin (the TARGET) | TomTom Traffic API | ⚠️ **SYNTHETIC placeholder** — must swap for real TomTom |
| **C — Road graph** | The road network (junctions + segments) | OpenStreetMap (OSMnx) | ⚠️ **HAND-CODED placeholder** — must swap for real OSMnx |

**Key insight on data:** Google Maps does NOT expose historical traffic speeds to
developers (data goes in, never comes out). **Use TomTom**, not Google, for Plane B.
TomTom's Traffic Stats API is purpose-built for "speed on every segment + event
impact analysis" and has a free tier (50k tile / 2.5k non-tile req/day, no credit
card) plus a 30-day MOVE portal trial for historical data.

---

## 4. What is REAL vs. PLACEHOLDER right now (critical to understand)

The pipeline runs end-to-end and produces correct results, BUT two pieces are
stand-ins because the sandbox where this was built blocked external APIs:

### PLACEHOLDER 1 — The road graph (`src/road_network.py`)
- **Currently:** 15 junctions + 24 segments hand-coded with real Bengaluru
  coordinates, lanes, and speeds.
- **Why:** the OSM Overpass API was network-blocked during development.
- **The fix:** replace the entire `JUNCTIONS` dict and `SEGMENTS` list with ONE
  OSMnx call. See §6 Correction 1. The hand-coded data is *faking* what OSMnx
  returns automatically (hundreds of real junctions/segments with full metadata).

### PLACEHOLDER 2 — The traffic speeds (`src/traffic_data.py`)
- **Currently:** synthetic fallback when no TomTom key; real TomTom backend
  (`fetch_tomtom_flow`, `snapshot_zone_speeds`, `load_tomtom_stats_export`) is
  stubbed and ready to finish (see INSTRUCTIONS.md Task B).
- **Why:** no traffic API access during development.
- **The fix:** provide a TomTom key and finish the loaders. Schema is identical,
  so the swap changes nothing downstream.

### FIXED — The event calendar (`src/events_calendar.py`)
- **Was:** hardcoded `EVENT_DAYS` dict of 6 fake events in `config.py`.
- **Now:** `derive_event_days()` builds the calendar from the REAL ASTraM CSV —
  22 event-days in the Chinnaswamy zone (148 city-wide), real dates, real snapped
  junctions, real start times. Crowd size is the only augmented field and always
  carries a `crowd_source` label. The hardcoded dict is removed from the active
  path.

**Everything else is real and correct:** the ASTraM loading/cleaning, the
graph-hop weighted impact, the baseline model logic, the delta computation, the
validation, the three output layers. The placeholders are isolated by design so
swapping them in changes nothing downstream.

---

## 5. Current architecture (file by file)

```
project_final/
├── data/
│   ├── raw/astram_events.csv          # REAL ASTraM event data
│   └── processed/                      # pipeline outputs (parquet)
├── models/                             # saved baseline model (joblib)
├── outputs/                            # impact reports (json)
├── requirements.txt
├── README.md
├── CLAUDE.md                           # this file
└── src/
    ├── config.py          # ALL tunable params (paths, features, hyperparams, event days)
    ├── road_network.py    # [PLACEHOLDER] builds road graph + seg_info. SWAP FOR OSMnx.
    ├── impact.py          # graph-hop weighted impact + betweenness centrality. REAL.
    ├── data_loader.py     # imports & cleans ASTraM, snaps events to junctions. REAL.
    ├── traffic_data.py    # [PLACEHOLDER] speed panel (Plane B). SWAP FOR TomTom.
    ├── features.py        # cyclical encoding, segment codes, non-event filter, time split. REAL.
    ├── baseline_model.py  # LightGBM baseline (normal-speed predictor) class. REAL.
    ├── delta.py           # delta computation + validation + 3 output layers. REAL.
    └── main.py            # orchestrates the full pipeline end to end.
```

**Run it:** `cd src && python main.py`
Each module also runs standalone for inspection (e.g. `python baseline_model.py`).

**Current verified results (on synthetic Plane B):**
- Baseline test MAE: ~1.17 km/h (predicts normal speed tightly)
- Non-event delta mean: ~0.00 (validation passes — baseline isn't biased)
- Cricket matches: ~20–21% congestion increase, worst segment −18 km/h
- Procession: ~4%, public event: ~7% (system differentiates event types correctly)

---

## 6. CORRECTIONS TO MAKE (in priority order)

### Correction 1 — Replace hand-coded graph with OSMnx [HIGH PRIORITY]
In `src/road_network.py`, replace `build_road_graph()` with:
```python
import osmnx as ox

def build_road_graph(venue_lat=VENUE_LAT, venue_lng=VENUE_LNG, radius_m=ZONE_RADIUS_M):
    G = ox.graph_from_point((venue_lat, venue_lng), dist=radius_m, network_type="drive")
    # Build seg_info from OSM edge attributes (lanes, maxspeed, length)
    seg_info = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        lanes = _parse_lanes(d.get("lanes", 2))
        ffs   = _parse_speed(d.get("maxspeed", 40))
        seg_info[f"{u}__{v}__{k}"] = {
            "u": u, "v": v,
            "road": d.get("name", "unnamed"),
            "lanes": lanes, "ffs": ffs,
            "capacity": lanes * ffs,
            "length_m": d.get("length", 0),
        }
    return G, seg_info
```
Add helper parsers `_parse_lanes` / `_parse_speed` (OSM values can be strings,
lists, or missing — handle gracefully with sensible defaults). Node lat/lng come
from `G.nodes[n]["y"]` (lat) and `["x"]` (lng). **The rest of the pipeline expects
`seg_info` with exactly these keys — keep them identical.**

### Correction 2 — Replace synthetic speeds with TomTom [HIGH PRIORITY]
In `src/traffic_data.py`, write a new `build_speed_panel()` that:
- For each segment, queries TomTom **Flow Segment Data** (real-time) on a 15-min
  schedule, OR pulls **Traffic Stats** historical area report from the MOVE portal.
- Returns a DataFrame with the EXACT existing schema:
  `segment_id, date, time, hour, weekday, is_weekend, is_rain, lanes,
   free_flow_speed, capacity, observed_speed, is_event_day, event_active, event_type`
- `is_event_day` / `event_active` / `event_type` come from joining the ASTraM
  events (Plane A) by date + spatial proximity, NOT from the API.
- Keep the synthetic generator available behind a flag (`USE_SYNTHETIC=True`) so the
  pipeline still runs without an API key.

### Correction 3 — Wire real weather into the baseline [MEDIUM]
Currently `is_rain` is synthetic. Pull real historical weather for Bengaluru
(OpenWeatherMap history or IMD) keyed by date, and join it so the baseline learns
the true rain effect. This sharpens delta attribution (rain vs. event).

### Correction 4 — Join real ASTraM events to event days [MEDIUM]
Currently `EVENT_DAYS` in `config.py` is a hand-curated dict of 6 events. Replace
with a function that derives event days directly from the cleaned ASTraM data
(filter gatherings/construction in-zone, group by date, attach junction + crowd
proxy). The crowd proxy can be regex-extracted from the `description` field
("1000 persons") with event-type priors as fallback.

### Correction 5 — Add a noise-floor guard on positive deltas [LOW]
In `delta.py`, large positive deltas on event days (event-day faster than baseline)
are real but rare; keep them but flag multi-event-overlap bins as `multi_event` and
exclude them from single-event training to avoid attribution confusion.

---

## 7. THE FULL FLOW (what the finished system does, stage by stage)

```
Stage 0  Build road graph (OSMnx) + centrality + hop-weights        [REAL logic, swap graph source]
Stage 1  Load & clean ASTraM events, snap to junctions              [DONE]
Stage 2  Get traffic speeds (TomTom) → segment×bin panel            [logic DONE, swap data source]
Stage 3  Engineer features (cyclical time, segment code, etc.)      [DONE]
Stage 4  Train BASELINE model on NON-EVENT data only                [DONE]
Stage 5  Compute DELTA = observed − baseline → impact footprint     [DONE]
─────────────────────────────────────────────────────────────────── (above = built)
Stage 6  DELTA PREDICTOR: forecast a FUTURE event's footprint       [TO BUILD]
Stage 7  SIMULATION (SUMO): inject event demand, toggle each
         intervention, MEASURE congestion removed (the proof)        [TO BUILD]
Stage 8  OPTIMIZER (OR-Tools): place officers/barricades/diversions
         optimally under resource limits, over simulated outcomes    [TO BUILD]
Stage 9  RECOMMENDATION: natural-language plan + additive waterfall
         (simulation-measured + SHAP-attributed, cross-validated)    [TO BUILD]
Stage 10 POST-EVENT LEARNING: predicted vs actual delta →
         recalibrate predictor, save reusable event templates        [TO BUILD]
```

---

## 8. WHAT TO BUILD NEXT (the roadmap)

**Next up: Stage 6 — the Delta Predictor.**
Train a model that maps `(event_type, junction, crowd, duration, hour, weekday,
weather, segment hop-weight, segment capacity, baseline_level) → predicted delta`,
using the deltas computed in Stage 5 as labels. This is what forecasts a future
event BEFORE it happens (planning mode — no observed value needed). Caveats to
respect: very few real event instances → pool event types in one model, output
confidence intervals (not bare point estimates), and plan to augment with
simulation-generated deltas.

**Then Stage 7 — SUMO simulation.** This is where "barricade removes X congestion"
becomes a MEASURED result, not a guess. Build the SUMO network from the same OSM
data, calibrate its no-event baseline to match real baseline speeds (gate
everything on this calibration check), inject event demand (crowd → vehicle trips
via mode-share), and script interventions via TraCI (officer = junction throughput
up, barricade = edge closed, diversion = edge cost change). Measure congestion per
scenario. The combined effect of multiple interventions MUST be simulated, not
summed — they interact non-linearly.

**Then Stage 8 — OR-Tools optimizer.** Officer placement as a Maximal Covering /
p-median problem over high-impact, high-centrality segments. Candidate diversions
from Yen's k-shortest-paths. Solve with CP-SAT. Search is GUIDED (footprint +
centrality prune the candidate space) — it is NOT blind brute-force; only the few
smart candidates get simulated.

**Then Stages 9–10** — LLM recommendation text (light, templating) + the post-event
learning loop (the centerpiece differentiator: show prediction error shrinking
across events).

---

## 9. KEY PRINCIPLES TO ENFORCE (judges will probe these)

1. **Baseline trained on non-event data ONLY.** Never let event traffic leak in —
   it would shrink deltas and under-count impact. This filter lives in
   `features.split_baseline_data` (`~is_event_day`). Protect it.
2. **Time-based train/test split, never random.** Random splits leak the future.
3. **No fabricated congestion target.** Delta is measured (observed − baseline) or
   simulated (SUMO), never invented by a regressor.
4. **Intervention effects are measured by simulation, not decomposed from a
   prediction.** You cannot split a predicted number into per-intervention shares;
   you BUILD UP each effect by toggling it in SUMO and measuring the difference.
5. **Combined effects are non-additive** — simulate the combination, don't sum singles.
6. **One city, deep.** Model the Chinnaswamy zone thoroughly. "Works for all India"
   is shallow; "works for this venue and the pipeline ports to any venue" is honest.
7. **Quantify uncertainty.** Report deltas with confidence intervals ("88 ± 6"),
   not bare numbers.

---

## 10. SCALABILITY STORY (for the presentation, and a real future feature)

The PIPELINE is city-agnostic; the DATA is local. To deploy to a new city (e.g.
Wankhede, Mumbai): give coordinates → OSMnx auto-builds the graph → pull TomTom
speeds for the zone → retrain baseline (minutes). All algorithms unchanged.

**Future-work upgrade (mention in pitch):** the baseline currently uses
`segment_code` (road identity) as a proxy for demand/context, so it can't
generalize to unseen roads. Replacing it with demand-side features (land-use, POI
density, population, road functional class — all from OSM) would let one model
generalize across cities → zero-shot deployment without local traffic history.
The delta predictor partially transfers already (event-impact patterns are
semi-universal), giving a warm start that the post-event loop calibrates locally =
cross-city transfer learning.

---

## 11. GOTCHAS / NOTES FOR FUTURE WORK

- ASTraM data is UTC; Bengaluru is UTC+5:30. Convert once, centrally. Already
  handled in `data_loader.parse_timestamps`.
- ASTraM is 94% UNPLANNED events, dominated by vehicle breakdowns (60%). The
  rally/festival "gathering" events are only ~327 in-city, ~21 in the Chinnaswamy
  zone. Be honest about this thinness; lean on simulation to augment.
- ASTraM has NO congestion measurement, NO attendance, NO deployment-outcome
  records. That's exactly why Plane B (TomTom) and simulation (SUMO) are required.
- 86 ASTraM events have real GPS route_paths (processions) — a unique feature few
  teams will use; model procession impact along the actual path, not a point.
- Keep `config.py` as the single source of truth for parameters. Don't scatter
  magic numbers in modules.
- TomTom historical trial is 30 days — pull the historical area report covering the
  ASTraM event dates EARLY and export it, so the data persists past the trial.
```
