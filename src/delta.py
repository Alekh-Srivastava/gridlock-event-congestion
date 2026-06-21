"""
============================================================================
DELTA — compute event impact, and produce the three output layers
============================================================================
delta = observed_speed - baseline_predicted_speed

  observed   = the REAL measured speed on the event day (from Plane B)
  baseline   = the model's prediction of normal speed (the counterfactual)
  delta < 0  => the event slowed traffic by |delta| km/h

From the per-segment per-bin deltas we build three output layers:
  1. FOOTPRINT       - segment x time heatmap (for the command room)
  2. PRIORITY_RANK   - segments by |delta| x centrality (for the optimizer)
  3. AGGREGATE       - one headline impact number (for executives)
"""
import pandas as pd

from config import NOISE_FLOOR_KMH, SEVERE_DELTA, MODERATE_DELTA


def compute_deltas(featured_panel, baseline_model):
    """
    Add baseline_predicted and delta columns to the full featured panel.
    Applies the noise floor: tiny deltas are zeroed as measurement noise.
    """
    df = featured_panel.copy()
    df["baseline_predicted"] = baseline_model.predict(df)
    df["delta"] = df["observed_speed"] - df["baseline_predicted"]

    # Noise floor — only on non-event bins, to avoid inventing phantom impact
    noise = df["delta"].abs() < NOISE_FLOOR_KMH
    df.loc[noise & ~df["event_active"], "delta_denoised"] = 0.0
    df["delta_denoised"] = df["delta_denoised"].fillna(df["delta"])
    return df


def validate_baseline(panel_with_deltas):
    """
    Sanity check: on NON-event days, delta should average ~0.
    If it doesn't, the baseline is biased. Returns a dict of checks.
    """
    non_event = panel_with_deltas[~panel_with_deltas["is_event_day"]]["delta"]
    return {
        "non_event_delta_mean": float(non_event.mean()),   # want ~0
        "non_event_delta_std":  float(non_event.std()),    # the noise floor
        "pct_beyond_5kmh":      float((non_event.abs() > 5).mean() * 100),
    }


def event_footprint(panel_with_deltas, event_date):
    """Layer 1: segment x time pivot of deltas for one event date."""
    day = panel_with_deltas[panel_with_deltas["date"] == event_date]
    return day.pivot_table(index="segment_id", columns="time",
                           values="delta", aggfunc="mean")


def priority_ranking(panel_with_deltas, event_date, seg_centrality, seg_info):
    """Layer 2: rank affected segments by |worst delta| x centrality."""
    ev = panel_with_deltas[
        (panel_with_deltas["date"] == event_date) & (panel_with_deltas["event_active"])
    ]
    if ev.empty:
        return pd.DataFrame()

    summary = ev.groupby("segment_id").agg(
        avg_delta=("delta", "mean"),
        worst_delta=("delta", "min"),
        bins_affected=("delta", "count"),
    ).reset_index()

    summary["road"]        = summary["segment_id"].map(lambda s: seg_info[s]["road"])
    summary["centrality"]  = summary["segment_id"].map(seg_centrality)
    summary["priority"]    = summary["worst_delta"].abs() * (1 + summary["centrality"] * 5)
    return summary.sort_values("priority", ascending=False).reset_index(drop=True)


def aggregate_impact(panel_with_deltas, event_date):
    """Layer 3: one headline impact summary for an event date."""
    ev = panel_with_deltas[
        (panel_with_deltas["date"] == event_date) & (panel_with_deltas["event_active"])
    ]
    if ev.empty:
        return None

    avg_baseline = ev["baseline_predicted"].mean()
    avg_delta    = ev["delta"].mean()
    return {
        "event_date":            event_date,
        "avg_delta_kmh":         round(avg_delta, 1),
        "worst_delta_kmh":       round(ev["delta"].min(), 1),
        "congestion_increase_pct": round(abs(avg_delta / avg_baseline) * 100, 1),
        "segments_affected":     int(ev["segment_id"].nunique()),
        "severe_segments":       int(ev[ev["delta"] < SEVERE_DELTA]["segment_id"].nunique()),
        "moderate_segments":     int(ev[(ev["delta"] < MODERATE_DELTA) &
                                        (ev["delta"] >= SEVERE_DELTA)]["segment_id"].nunique()),
        "impact_window":         (ev["time"].min(), ev["time"].max()),
    }


if __name__ == "__main__":
    from road_network import build_road_graph
    from traffic_data import build_speed_panel
    from features import build_feature_table, split_baseline_data
    from baseline_model import BaselineModel
    from impact import segment_centrality

    G, seg_info = build_road_graph()
    panel = build_speed_panel(G, seg_info)
    featured, cats = build_feature_table(panel)
    split = split_baseline_data(featured)

    bm = BaselineModel(); bm.train(split, cats)
    panel_d = compute_deltas(featured, bm)

    print("Validation:", validate_baseline(panel_d))
    agg = aggregate_impact(panel_d, "2024-03-09")
    print("\nIPL 2024-03-09 impact:", agg)
