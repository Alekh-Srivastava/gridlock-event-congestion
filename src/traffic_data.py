
"""
============================================================================
TRAFFIC DATA — the speed panel (Plane B)
============================================================================
Provides observed speed per segment per 15-min bin. Two backends:

  1. TomTom  (REAL)  — use on your machine with an API key.
  2. Synthetic (FALLBACK) — only when no API access (e.g. sandbox).

Both return the SAME schema, so the rest of the pipeline is agnostic.
Set use_synthetic=False + pass a tomtom_key to use real data.
"""
import time
import numpy as np
import pandas as pd

from config import (BINS_PER_DAY, BIN_MINUTES, SYNTH_N_DAYS, SYNTH_START,
                    SYNTH_RAIN_FRAC, SYNTH_SEED, SYNTH_MAX_SEGMENTS,
                    VENUE_LAT, VENUE_LNG,
                    HOP_DECAY, MAX_HOPS, CAPACITY_ADJUSTMENT)
from impact import hop_distance_map
from road_network import haversine_m

try:
    import requests
except ImportError:
    requests = None

TOMTOM_FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"


# ----- BACKEND 1: REAL TomTom -----
def fetch_tomtom_flow(lat, lng, api_key):
    """Query TomTom Flow Segment Data for one point (real-time speed)."""
    if requests is None:
        raise ImportError("pip install requests to use TomTom")
    try:
        resp = requests.get(TOMTOM_FLOW_URL,
                            params={"point": f"{lat},{lng}", "key": api_key, "unit": "KMPH"},
                            timeout=10)
        resp.raise_for_status()
        d = resp.json()["flowSegmentData"]
        return {"current_speed": d["currentSpeed"],
                "free_flow_speed": d["freeFlowSpeed"],
                "confidence": d.get("confidence")}
    except Exception as e:
        print(f"  [tomtom] failed ({lat:.4f},{lng:.4f}): {e}")
        return None


def snapshot_zone_speeds(G, seg_info, api_key, sleep_s=0.2):
    """One real-time speed snapshot for every segment. Run on a 15-min cron
    to accumulate a real historical panel."""
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    rows = []
    for sid, info in seg_info.items():
        lat = (G.nodes[info["u"]]["lat"] + G.nodes[info["v"]]["lat"]) / 2
        lng = (G.nodes[info["u"]]["lng"] + G.nodes[info["v"]]["lng"]) / 2
        flow = fetch_tomtom_flow(lat, lng, api_key)
        if flow:
            rows.append({"segment_id": sid, "date": now.strftime("%Y-%m-%d"),
                         "time": now.strftime("%H:%M"), "hour": now.hour + now.minute/60,
                         "weekday": now.day_name(), "is_weekend": now.weekday() >= 5,
                         "lanes": info["lanes"], "free_flow_speed": flow["free_flow_speed"],
                         "capacity": info["capacity"], "observed_speed": flow["current_speed"]})
        time.sleep(sleep_s)
    return pd.DataFrame(rows)


def load_tomtom_stats_export(csv_path):
    """Load a historical export from the TomTom Traffic Stats (MOVE) portal.
    Rename its columns to our schema once you see the actual export."""
    df = pd.read_csv(csv_path)
    # TODO: map columns to: segment_id, date, time, hour, weekday, is_weekend,
    #       lanes, free_flow_speed, capacity, observed_speed
    return df


# ----- BACKEND 2: SYNTHETIC fallback -----
def _baseline_speed(hour, ffs, capacity, is_weekend, is_rain, rng):
    mr = max(0, 1 - abs(hour - 9) / 2.5) * (12 if not is_weekend else 4)
    er = max(0, 1 - abs(hour - 18) / 3) * (15 if not is_weekend else 6)
    rp = ffs * 0.15 if is_rain else 0
    nb = max(0, 1 - abs(hour - 3) / 4) * 8 if (hour < 6 or hour > 22) else 0
    cp = (1 - capacity / 160) * 5
    speed = ffs - mr - er - rp + nb - cp + rng.normal(0, 1.5)
    return float(np.clip(speed, 5, ffs + 5))


def _event_delta(hour, event, weight, rng):
    if weight <= 0: return 0.0
    es, ee = event["start_h"], event["end_h"]; peak = es + 1.5
    if not (es - 1 <= hour <= ee + 1): return 0.0
    if hour < es:      tf = max(0, 1 - (es - hour))
    elif hour <= peak: tf = min(1, (hour - es) / (peak - es))
    elif hour <= ee:   tf = max(0.5, 1 - (hour - peak) / (ee - peak) * 0.5)
    else:              tf = max(0, 0.5 * (1 - (hour - ee)))
    return max(0.0, weight * tf * (25 * event["crowd"] / 40000) + rng.normal(0, 0.8))


def _precompute_event_weights(G, seg_info, event_days):
    max_cap = max(s["capacity"] for s in seg_info.values())
    out = {}
    for ev in event_days.values():
        jn = ev["junction"]
        if jn in out: continue
        hop_map = hop_distance_map(G, jn)
        w = {}
        for sid, info in seg_info.items():
            mh = min(hop_map.get(info["u"], 999), hop_map.get(info["v"], 999))
            if mh <= MAX_HOPS:
                w[sid] = HOP_DECAY.get(mh, 0) * (1 - CAPACITY_ADJUSTMENT * info["capacity"] / max_cap)
        out[jn] = w
    return out


def _nearest_segments(G, seg_info, n, anchor_nodes=None):
    """Return the N segments whose midpoint is geographically closest to the
    venue OR to any event junction (anchor_nodes). Anchoring on event junctions
    ensures segments affected by each event are always included in synthetic mode,
    even when one-way street topology makes those junctions many hops from the
    venue-centric cluster.
    """
    def _min_dist_to_anchors(info):
        lat = (G.nodes[info["u"]]["lat"] + G.nodes[info["v"]]["lat"]) / 2
        lng = (G.nodes[info["u"]]["lng"] + G.nodes[info["v"]]["lng"]) / 2
        d_venue = haversine_m(VENUE_LAT, VENUE_LNG, lat, lng)
        if not anchor_nodes:
            return d_venue
        d_anchors = min(
            haversine_m(G.nodes[a]["lat"], G.nodes[a]["lng"], lat, lng)
            for a in anchor_nodes
        )
        return min(d_venue, d_anchors)

    ranked = sorted(seg_info.items(), key=lambda kv: _min_dist_to_anchors(kv[1]))
    return dict(ranked[:n])


def build_synthetic_panel(G, seg_info, event_days):
    """Synthetic speeds — fallback only. event_days comes from the REAL ASTraM
    CSV, so WHEN/WHERE/TYPE of events is real; only speed numbers are simulated."""
    print("[traffic_data] SYNTHETIC speed panel (fallback — no TomTom access).")

    # Cap segments to avoid OOM: real OSMnx graphs have 6,000+ segments;
    # full synthetic panel would be ~95M rows. Keep the N nearest to the venue OR
    # to any crowd-generating event junction (so every real event gets coverage).
    if len(seg_info) > SYNTH_MAX_SEGMENTS:
        crowd_junctions = list({
            ev["junction"] for ev in event_days.values()
            if ev.get("crowd", 0) > 0 and ev["junction"] in G.nodes
        })
        seg_info = _nearest_segments(G, seg_info, SYNTH_MAX_SEGMENTS,
                                     anchor_nodes=crowd_junctions)
        print(f"[traffic_data] Synthetic mode: using {SYNTH_MAX_SEGMENTS} nearest "
              f"segments (venue + {len(crowd_junctions)} event junctions as anchors; "
              f"real TomTom will use all {G.number_of_edges():,}).")

    rng = np.random.default_rng(SYNTH_SEED)
    dates = pd.date_range(SYNTH_START, periods=SYNTH_N_DAYS, freq="D")
    bin_times = pd.date_range("00:00", periods=BINS_PER_DAY, freq=f"{BIN_MINUTES}min").time
    hours = np.array([t.hour + t.minute / 60 for t in bin_times])   # (N_B,)
    time_strs = [t.strftime("%H:%M") for t in bin_times]
    print(f"[traffic_data] Synthetic panel: {SYNTH_N_DAYS} days, {BINS_PER_DAY} bins/day,  nearest segments function is passed without crash ")
    # Index-based rain sampling avoids datetime64 vs Timestamp hash mismatch
    rain_idx = rng.choice(SYNTH_N_DAYS, size=int(SYNTH_N_DAYS * SYNTH_RAIN_FRAC), replace=False)
    rain_days_set = set(rain_idx.tolist())

    ev_weights = _precompute_event_weights(G, seg_info, event_days)
    print(f"[traffic_data] Precomputed event weights for {len(ev_weights)} junctions.")   
    seg_ids = list(seg_info.keys())
    n_seg  = len(seg_ids)
    n_days = SYNTH_N_DAYS
    n_bins = BINS_PER_DAY

    ffs_arr   = np.array([seg_info[sid]["ffs"]      for sid in seg_ids], dtype=float)
    cap_arr   = np.array([seg_info[sid]["capacity"]  for sid in seg_ids], dtype=float)
    lanes_arr = np.array([seg_info[sid]["lanes"]     for sid in seg_ids], dtype=int)

    is_weekend_arr = np.array([d.weekday() >= 5 for d in dates], dtype=bool)
    is_rain_arr    = np.array([i in rain_days_set for i in range(n_days)], dtype=bool)
    date_strs      = [d.strftime("%Y-%m-%d") for d in dates]

    # --- Vectorised baseline: shape (N_S, N_D, N_B) --------------------------
    # Pre-generate all noise up-front — avoids 7.2M per-row rng calls
    noise_b = rng.normal(0, 1.5, (n_seg, n_days, n_bins))
    noise_d = rng.normal(0, 0.8, (n_seg, n_days, n_bins))

    h = hours  # (N_B,)
    mr_factor = np.where(is_weekend_arr, 4.0, 12.0)   # (N_D,)
    er_factor = np.where(is_weekend_arr, 6.0, 15.0)   # (N_D,)

    mr = (np.maximum(0, 1 - np.abs(h - 9)  / 2.5)[np.newaxis, np.newaxis, :]
          * mr_factor[np.newaxis, :, np.newaxis])  # (1, N_D, N_B)
    er = (np.maximum(0, 1 - np.abs(h - 18) / 3.0)[np.newaxis, np.newaxis, :]
          * er_factor[np.newaxis, :, np.newaxis])  # (1, N_D, N_B)
    nb = np.where((h < 6) | (h > 22),
                  np.maximum(0, 1 - np.abs(h - 3) / 4) * 8, 0.0)  # (N_B,)
    rp = (ffs_arr[:, np.newaxis, np.newaxis]
          * 0.15 * is_rain_arr[np.newaxis, :, np.newaxis])          # (N_S, N_D, 1)
    cp = (1 - cap_arr / 160) * 5                                    # (N_S,)

    baselines = np.clip(
        ffs_arr[:, np.newaxis, np.newaxis]
        - mr - er - rp
        + nb[np.newaxis, np.newaxis, :]
        - cp[:, np.newaxis, np.newaxis]
        + noise_b,
        5.0,
        ffs_arr[:, np.newaxis, np.newaxis] + 5.0,
    )  # (N_S, N_D, N_B)

    # --- Vectorised event deltas: loop only over event days (~22 iters) ------
    event_deltas = np.zeros((n_seg, n_days, n_bins))

    for d_idx, ds in enumerate(date_strs):
        event = event_days.get(ds)
        if event is None:
            continue
        seg_w = ev_weights.get(event["junction"], {})
        if not seg_w:
            continue
        crowd = event["crowd"]
        if crowd == 0:
            continue  # construction/lane-closure events have no crowd signal; skip
        es, ee   = float(event["start_h"]), float(event["end_h"])
        peak     = es + 1.5
        crowd_f  = 25 * crowd / 40000

        in_window = (h >= es - 1) & (h <= ee + 1)  # (N_B,)
        tf = np.where(
            h < es,
            np.maximum(0, 1 - (es - h)),
            np.where(h <= peak,
                     np.minimum(1, (h - es) / max(peak - es, 1e-9)),
                     np.where(h <= ee,
                              np.maximum(0.5, 1 - (h - peak) / max(ee - peak, 1e-9) * 0.5),
                              np.maximum(0, 0.5 * (1 - (h - ee))))))
        tf = tf * in_window  # (N_B,)

        w_arr = np.array([seg_w.get(sid, 0.0) for sid in seg_ids])  # (N_S,)
        delta = np.maximum(
            0.0,
            w_arr[:, np.newaxis] * tf[np.newaxis, :] * crowd_f
            + noise_d[:, d_idx, :],
        )  # (N_S, N_B)
        event_deltas[:, d_idx, :] = delta

    observed = np.maximum(3.0, baselines - event_deltas)  # (N_S, N_D, N_B)
    active   = event_deltas > 0.5                          # (N_S, N_D, N_B)

    # --- Build DataFrame from index arrays (no row-level Python loop) --------
    is_event_day_arr   = np.array([ds in event_days for ds in date_strs], dtype=bool)
    event_type_per_day = [event_days[ds]["type"] if ds in event_days else "none"
                          for ds in date_strs]

    seg_col        = np.repeat(seg_ids, n_days * n_bins)
    date_col       = np.tile(np.repeat(date_strs, n_bins), n_seg)
    time_col       = np.tile(time_strs, n_seg * n_days)
    hour_col       = np.tile(np.round(hours, 2), n_seg * n_days)
    weekday_col    = np.tile(np.repeat([d.day_name() for d in dates], n_bins), n_seg)
    is_weekend_col = np.tile(np.repeat(is_weekend_arr, n_bins), n_seg)
    is_rain_col    = np.tile(np.repeat(is_rain_arr, n_bins), n_seg)
    lanes_col      = np.repeat(lanes_arr, n_days * n_bins)
    ffs_col        = np.repeat(ffs_arr,   n_days * n_bins)
    cap_col        = np.repeat(cap_arr,   n_days * n_bins)
    obs_col        = np.round(observed.ravel(), 1)
    is_evt_day_col = np.tile(np.repeat(is_event_day_arr, n_bins), n_seg)
    active_col     = active.ravel()
    evt_type_base  = np.tile(np.repeat(event_type_per_day, n_bins), n_seg)
    evt_type_col   = np.where(active_col, evt_type_base, "none")

    panel = pd.DataFrame({
        "segment_id":     seg_col,
        "date":           date_col,
        "time":           time_col,
        "hour":           hour_col,
        "weekday":        weekday_col,
        "is_weekend":     is_weekend_col,
        "is_rain":        is_rain_col,
        "lanes":          lanes_col,
        "free_flow_speed": ffs_col,
        "capacity":       cap_col,
        "observed_speed": obs_col,
        "is_event_day":   is_evt_day_col,
        "event_active":   active_col,
        "event_type":     evt_type_col,
    })
    print(f"[traffic_data] Synthetic panel: {len(panel):,} rows.")
    return panel


def build_speed_panel(G, seg_info, event_days, use_synthetic=True, tomtom_key=None):
    """Unified entry. Default synthetic (sandbox-safe). On your machine pass
    use_synthetic=False + tomtom_key for real data."""
    if use_synthetic or not tomtom_key:
        return build_synthetic_panel(G, seg_info, event_days)
    raise NotImplementedError(
        "Accumulate snapshot_zone_speeds() on a schedule, or load_tomtom_stats_export() "
        "from a MOVE export, then join ASTraM event flags. See CLAUDE.md Correction 2.")
