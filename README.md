# GridLock — AI-Powered Event Traffic Intelligence

**Team Srikrit · Flipkart GRID 2026 · Theme 2: Event-Driven Congestion**

> *When a cricket match starts at 7pm Saturday at Chinnaswamy Stadium, which roads slow down, by how much, and where should officers be deployed — answered before the event begins.*

---

## The Problem

Bengaluru hosts hundreds of major events every year. Every rally, cricket match, or procession triggers cascading traffic jams — but no system today quantifies the impact **before** it happens:

- **Events cause unpredictable cascades.** A match at Chinnaswamy doesn't just slow M.G. Road — it backs up Residency, Brigade, and Richmond in a wave that spreads outward. Police see it only after it's too late to stop.
- **Officers are deployed reactively.** No system tells a superintendent *before* the event: "put 4 officers here, close this road, reroute via MES Circle."
- **Existing tools measure the past.** ASTraM logs incidents after they occur. Microsimulation is generic. The decision layer between "event announced" and "officers deployed" doesn't exist — GridLock builds it.

---

## The Core Idea

```
ML predicts the PROBLEM.          Simulation proves the SOLUTION.
(how much congestion will happen)  (how much each intervention removes)
```

Congestion is never fabricated. It is measured as a **delta**:

```
δ = observed_speed − baseline_speed
    (real, measured)   (model's "no-event" counterfactual)
```

The baseline is trained **only on non-event data** — it already accounts for rush hour, rain, and weekday. So `δ < 0` is purely the event's contribution. A judge asking *"how do you know the event caused this and not rush hour?"* gets a precise answer: *"the baseline already removes rush hour — the delta is what's left."*

---

## Two LightGBM Models

| | Model 1 — Baseline | Model 2 — Delta Predictor |
|---|---|---|
| **Purpose** | Predict normal (no-event) speed | Predict event impact before it happens |
| **Trained on** | Non-event rows only (`is_event_day = False`) | Event-active rows with measured deltas as labels |
| **Input** | `segment_code, time, weekday, weather, road` | `hop_from_event, crowd, event_type, road_capacity, time_of_day` |
| **Output** | `baseline_speed` (km/h) | `δ` (km/h) with 87% confidence interval |
| **Training rows** | ~1.4M (150 days × all segments) | ~3,800 (22 event-days × affected segments) |

**Key feature — `hop_from_event`:** BFS graph-hops from the event junction to each road segment. Roads 1 hop away feel maximum impact; roads 4+ hops away barely feel it. This spatial decay is what makes predictions geographically meaningful — not "the city slows down" but "these specific roads slow by this much."

---

## Full 12-Stage Pipeline

| Stage | What it does | Status |
|------:|--------------|--------|
| 0 | Build road graph (OSMnx) + betweenness centrality + hop-weights | ✅ |
| 1 | Load & clean ASTraM events, snap to junctions | ✅ |
| 2 | Traffic speed panel — TomTom live (synthetic fallback) | ✅ |
| 3 | Feature engineering — cyclical time encoding, segment codes | ✅ |
| 4 | Train Baseline Model (LightGBM #1) on non-event data only | ✅ |
| 5 | Compute deltas + validate baseline is unbiased | ✅ |
| 6 | Per-event impact reports — footprint + priority ranking | ✅ |
| 7 | Train Delta Predictor (LightGBM #2) on measured deltas | ✅ |
| 8 | Predict congestion footprint for a future event | ✅ |
| 9 | BPR simulation — measure congestion removed per intervention | ✅ |
| 10 | Greedy submodular optimizer — optimal officer/barricade placement | ✅ |
| 11 | Natural-language deployment brief | ✅ |
| 12 | Post-event learning loop — RMSE improves across events | ✅ |

**Verified results (on real ASTraM events, synthetic Plane B):**
- Baseline test MAE: ~1.17 km/h
- Non-event delta mean: ~0.00 (baseline is unbiased ✅)
- Post-event learning: RMSE 1.83 → 1.48 km/h (19% improvement over 5 measured events)

---

## Three Data Planes

| Plane | What | Source | Status |
|-------|------|--------|--------|
| **A — Events** | 8,173 incident records, Nov 2023–Apr 2024 | Bengaluru ASTraM (BTP) | ✅ Real |
| **B — Traffic** | Speed per segment per 15-min bin | TomTom Flow API | ✅ Live (synthetic fallback active) |
| **C — Road graph** | Junctions, segments, lanes, speed limits | OpenStreetMap (OSMnx) | ✅ Real |

**TomTom integration:** `fetch_tomtom_flow()` in `traffic_data.py` is fully implemented. Set `USE_SYNTHETIC = False` and add your API key in `config.py` — nothing else changes. The synthetic fallback generates statistically realistic speeds (rush-hour peaks, weekend dips, rain coefficients) so the pipeline runs without a key.

**ASTraM data:** 22 event-days in the Chinnaswamy zone (148 city-wide). 86 events have GPS route paths for processions. Crowd size is the only augmented field, always labelled with its source.

---

## Project Structure

```
event_congestion_project/
├── app.py                    # Full analysis dashboard (Streamlit)
├── app_demo.py               # Guided video demo — 8 stages unlock step by step
├── requirements.txt
├── data/
│   ├── raw/astram_events.csv # Real ASTraM data (8,173 events)
│   └── processed/            # Pipeline outputs (parquet)
├── models/                   # Saved LightGBM models (joblib)
├── outputs/                  # Impact reports, learning history (JSON)
└── src/
    ├── config.py             # All tunable parameters (single source of truth)
    ├── road_network.py       # OSMnx road graph builder
    ├── data_loader.py        # ASTraM import & cleaning
    ├── events_calendar.py    # Derives real event calendar from ASTraM CSV
    ├── traffic_data.py       # TomTom loader + synthetic fallback
    ├── features.py           # Cyclical encoding, segment codes, train/test split
    ├── baseline_model.py     # LightGBM #1 (normal-speed predictor)
    ├── delta.py              # Delta computation, validation, 3 output layers
    ├── impact.py             # Betweenness centrality, hop-weighted impact
    ├── delta_predictor.py    # LightGBM #2 (event impact forecaster)
    ├── simulation.py         # BPR volume-delay simulation
    ├── optimizer.py          # Greedy submodular optimizer
    ├── recommender.py        # Natural-language deployment brief
    ├── post_event_learning.py# Chronological learning loop
    └── main.py               # Full pipeline orchestrator
```

---

## Running the Project

### Full pipeline (headless)

```bash
pip install -r requirements.txt
cd src
python main.py
```

### Interactive demo app (recommended for video)

```bash
pip install -r requirements.txt
streamlit run app_demo.py
```

Opens at `http://localhost:8501`. Eight stages unlock step by step — enter any coordinates, upload an events CSV (or use the bundled ASTraM data), and walk through the full system live.

### Full dashboard

```bash
streamlit run app.py
```

---

## Key Design Decisions

**Non-event filter is the core correctness guarantee.** If event-day traffic leaks into baseline training, the model learns "slow roads are normal" — shrinking deltas and undercounting impact. The filter `is_event_day = False` in `features.split_baseline_data()` is the single most important line in the codebase.

**Time-based train/test split, never random.** Random splits leak future traffic patterns into training. We enforce a chronological cutoff so the test set is genuinely unseen.

**Priority score = |Δ km/h| × (1 + centrality × 5).** Betweenness centrality (BC) measures how many shortest paths pass through a road. A moderate slowdown on a high-BC bottleneck road cascades further than a large slowdown on a quiet side street. Officers go to bottleneck junctions — not just the worst-delta road.

**Intervention effects are simulated, not summed.** BPR volume-delay model (`t = t₀ × (1 + 0.15 × (V/C)⁴)`) simulates each combination. Combined effects are non-additive — two diversions can interact. We build them up with the greedy submodular optimizer (≥ 63% of global optimum guaranteed by (1 − 1/e) submodularity bound).

**Confidence intervals on every prediction.** The Delta Predictor reports `[δ_lo, δ_hi]` at 87% CI using per-event-type residual std from the training set. A rarely-seen event type gives a wide CI; a well-observed type gives a tight one.

---

## Scalability

The **pipeline is city-agnostic; the data is local.** To deploy to a new venue (e.g. Wankhede, Mumbai):

1. Enter new coordinates → OSMnx auto-builds the road graph
2. Pull TomTom speeds for the zone
3. Retrain baseline (minutes) — algorithms unchanged

**Future path:** replace `segment_code` (road identity) with demand-side features (land-use, POI density, road class from OSM). A model trained on one city then generalises to any city — zero-shot deployment without local traffic history. The Delta Predictor partially transfers already (event-impact patterns are semi-universal), giving a warm start that the post-event loop calibrates locally.

---

## Post-Event Learning Loop

After each real event, the system:
1. Predicts the footprint using the current model (before the event)
2. Compares prediction to actual delta from the speed panel
3. Computes RMSE for this event
4. Retrains the Delta Predictor on all accumulated events
5. Saves the event as a reusable template

**Result:** RMSE 1.83 → 1.48 km/h across 5 measured events — 19% improvement. Each new real event meaningfully improves accuracy. This is the learning flywheel that makes the system more accurate over time.

---

*Built with Python · LightGBM · OSMnx · Streamlit · PyDeck · Altair*
