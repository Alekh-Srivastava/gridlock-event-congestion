"""
============================================================================
STAGE 9 — RECOMMENDATION GENERATOR
============================================================================
Converts the optimizer output into a human-readable operational deployment
brief that a traffic police commander can act on immediately.

Design: template-based (no LLM dependency for the prototype). The numbers
come entirely from Stages 6-8 — the text just narrates them. An LLM call
can replace the template strings later without changing anything else.

Output structure:
  1. Situation summary     — event, expected footprint, worst segments
  2. Deployment plan       — WHERE, WHAT, WHEN for each resource
  3. Expected improvement  — before/after numbers from simulation
  4. Confidence note       — honest statement about data basis
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _severity_label(delta_kmh):
    if delta_kmh < -10:
        return "SEVERE"
    elif delta_kmh < -5:
        return "MODERATE"
    elif delta_kmh < -2:
        return "MILD"
    return "MINIMAL"


def _format_time(hour_float):
    h = int(hour_float)
    m = int((hour_float % 1) * 60)
    return f"{h:02d}:{m:02d}"


def _top_roads(footprint_df, n=5):
    """Return the n worst-affected road names from the footprint."""
    agg = (
        footprint_df
        .groupby("road")["delta_pred"]
        .min()
        .sort_values()
        .head(n)
    )
    return list(agg.index), list(agg.values)


def _pre_position_time(start_h, offset_h=1.5):
    """Suggest deployment time = event_start − offset_h."""
    pre_h = max(0, start_h - offset_h)
    return _format_time(pre_h)


# ---------------------------------------------------------------------------
# Main recommendation function
# ---------------------------------------------------------------------------
def generate_recommendation(event_spec, footprint_df, optimal_plan,
                              opt_result, sim_before, sim_after,
                              seg_info=None, event_date=None):
    """
    Build the full operational brief as a formatted string.

    Parameters
    ----------
    event_spec   : dict from Stage 6 (type, crowd, start_h, end_h, ...)
    footprint_df : DataFrame from Stage 6 (predicted per-segment deltas)
    optimal_plan : list of intervention dicts from Stage 8
    opt_result   : summary dict from Stage 8 optimizer.report()
    sim_before   : {segment_id: speed_kmh}  no-intervention scenario
    sim_after    : {segment_id: speed_kmh}  with all interventions
    seg_info     : optional, for road names
    event_date   : optional date string for the brief header
    """
    ev_type   = str(event_spec.get("type", "event")).replace("_", " ").title()
    crowd     = int(event_spec.get("crowd", 0))
    start_h   = float(event_spec.get("start_h", 18.0))
    end_h     = float(event_spec.get("end_h", 22.0))
    weekday   = str(event_spec.get("weekday", ""))
    deploy_t  = _pre_position_time(start_h)
    is_rain   = bool(event_spec.get("is_rain", False))

    # Footprint summary
    if not footprint_df.empty:
        worst_delta = float(footprint_df["delta_pred"].min())
        avg_delta   = float(footprint_df["delta_pred"].mean())
        n_affected  = int(footprint_df["segment_id"].nunique())
        ci_lo       = float(footprint_df["delta_lo"].min())
        ci_hi       = float(footprint_df["delta_hi"].min())
        top_roads, top_deltas = _top_roads(footprint_df)
        severity = _severity_label(worst_delta)
    else:
        worst_delta = avg_delta = 0.0
        n_affected  = 0
        ci_lo = ci_hi = 0.0
        top_roads, top_deltas = [], []
        severity = "UNKNOWN"

    before_kmh = opt_result.get("avg_speed_before", 0.0)
    after_kmh  = opt_result.get("avg_speed_after",  0.0)
    gain_kmh   = opt_result.get("net_improvement_kmh", 0.0)

    # Approximate delay reduction (assuming 1 km corridor)
    delay_before_min = round(1.0 / max(before_kmh, 1) * 60, 1) if before_kmh > 0 else "?"
    delay_after_min  = round(1.0 / max(after_kmh,  1) * 60, 1) if after_kmh  > 0 else "?"

    # Count resource types
    n_officers   = sum(1 for iv in optimal_plan if iv["type"] == "officer")
    n_barricades = sum(1 for iv in optimal_plan if iv["type"] == "barricade")
    n_diversions = sum(1 for iv in optimal_plan if iv["type"] == "diversion")

    # Confidence interval text
    ci_text = (f"{worst_delta:+.1f} km/h  [{ci_lo:+.1f}, {ci_hi:+.1f}] 87% CI"
               if footprint_df is not None and not footprint_df.empty else "N/A")

    lines = []
    lines.append("=" * 72)
    lines.append("  TRAFFIC DEPLOYMENT BRIEF")
    if event_date:
        lines.append(f"  Event Date : {event_date}  ({weekday})")
    lines.append(f"  Event Type : {ev_type}")
    lines.append(f"  Crowd      : ~{crowd:,} persons  |  "
                 f"Window: {_format_time(start_h)} – {_format_time(end_h)}")
    if is_rain:
        lines.append("  Weather    : RAIN FORECAST — rain adds ~15% baseline congestion")
    lines.append("=" * 72)

    lines.append("\n  PREDICTED IMPACT (before intervention)")
    lines.append(f"  Severity          : {severity}")
    lines.append(f"  Worst segment Δ   : {ci_text}")
    lines.append(f"  Avg corridor Δ    : {avg_delta:+.1f} km/h across {n_affected} segments")
    if top_roads:
        lines.append(f"  Top affected roads:")
        for road, delta in zip(top_roads[:5], top_deltas[:5]):
            bar = "█" * max(1, int(abs(delta) / 2))
            lines.append(f"    {road:<32} {delta:+.1f} km/h  {bar}")

    lines.append("\n  DEPLOYMENT PLAN")
    lines.append(f"  Pre-position all resources by: {deploy_t}")
    lines.append(f"  Resources: {n_officers} officers | "
                 f"{n_barricades} barricades | {n_diversions} diversions")
    lines.append("")
    for i, iv in enumerate(optimal_plan, 1):
        iv_type  = iv.get("type", "").upper()
        label    = iv.get("label", "")
        if iv_type == "OFFICER":
            action = f"Deploy traffic officer — direct and optimize flow"
        elif iv_type == "BARRICADE":
            action = f"Place barricade — close segment, force diversion"
        else:
            action = f"Install diversion signage — redirect 50% flow"
        lines.append(f"  [{i}] {iv_type:<10}  {label}")
        lines.append(f"       Action: {action}")
        lines.append(f"       By    : {deploy_t}  (≥ {int(start_h - float(deploy_t.split(':')[0]) - float(deploy_t.split(':')[1])/60):.0f}h before event start)")

    lines.append("\n  SIMULATED OUTCOME (BPR traffic model)")
    lines.append(f"  Avg corridor speed WITHOUT intervention : {before_kmh:.1f} km/h")
    lines.append(f"  Avg corridor speed WITH intervention    : {after_kmh:.1f} km/h")
    lines.append(f"  Net speed improvement                   : +{gain_kmh:.1f} km/h")
    lines.append(f"  Estimated delay (1 km) WITHOUT          : {delay_before_min} min")
    lines.append(f"  Estimated delay (1 km) WITH             : {delay_after_min} min")

    lines.append("\n  DATA BASIS & CONFIDENCE")
    lines.append("  · Impact forecast: LightGBM delta predictor trained on real ASTraM events")
    lines.append("  · Road network   : OpenStreetMap (OSMnx), real Bengaluru topology")
    lines.append("  · Traffic model  : BPR volume-delay (α=0.15, β=4, standard calibration)")
    lines.append("  · Crowd size     : " + (
        "from ASTraM description text"
        if event_spec.get("crowd_source") == "text"
        else f"estimated ({event_spec.get('crowd_source','prior')} — see events_calendar.py)"
    ))
    lines.append("  · Optimization   : Greedy submodular coverage ≥ 63% of global optimum")
    lines.append("  · NOTE           : Plane B (traffic speeds) is synthetic pending TomTom.")
    lines.append("    Predicted deltas will sharpen with real historical speed data.")
    lines.append("=" * 72)

    return "\n".join(lines)


def generate_post_event_note(event_spec, predicted_delta, actual_delta):
    """
    Short note for the learning loop (Stage 10): actual vs predicted.
    Printed after the event to track model calibration over time.
    """
    err = actual_delta - predicted_delta
    return (
        f"\n  [POST-EVENT] {event_spec.get('type','event')}  "
        f"Predicted Δ: {predicted_delta:+.1f} km/h  |  "
        f"Actual Δ: {actual_delta:+.1f} km/h  |  "
        f"Error: {err:+.1f} km/h"
    )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from road_network import build_road_graph
    from data_loader import load_clean_events
    from events_calendar import derive_event_days
    from traffic_data import build_speed_panel
    from features import build_feature_table, split_baseline_data
    from baseline_model import BaselineModel
    from delta import compute_deltas
    from delta_predictor import DeltaPredictor, build_predictor_training_data
    from impact import segment_centrality
    from simulation import EventSimulator
    from optimizer import generate_candidates, InterventionOptimizer

    G, seg_info    = build_road_graph()
    events         = load_clean_events(zone_only=True, G=G)
    event_days     = derive_event_days(events, zone_only=True)
    panel          = build_speed_panel(G, seg_info, event_days)
    featured, cats = build_feature_table(panel)
    split          = split_baseline_data(featured)
    bm             = BaselineModel(); bm.train(split, cats)
    panel_d        = compute_deltas(featured, bm)
    seg_bc         = segment_centrality(G, seg_info)

    train_df = build_predictor_training_data(panel_d, event_days, G, seg_info)
    dp = DeltaPredictor(); dp.train(train_df)

    crowd_events = {d: ev for d, ev in event_days.items() if ev["crowd"] > 0}
    date_str, ev = next(iter(crowd_events.items()))
    jn = ev["junction"]

    if jn not in G.nodes:
        print("Junction not in graph — skipping recommender test.")
    else:
        event_spec = {
            **ev,
            "type": ev["type"], "junction": jn,
            "weekday": "Saturday", "is_weekend": True, "is_rain": False,
        }
        fp         = dp.predict_footprint(event_spec, G, seg_info, bm, cats)
        candidates = generate_candidates(fp, seg_bc, seg_info, G)

        sim  = EventSimulator(G, seg_info)
        opt  = InterventionOptimizer(sim)
        plan, summary, before, after = opt.optimize(
            event_spec, fp, candidates, n_officers=4, n_barricades=2, n_diversions=1
        )
        result = opt.report(plan, summary, before, after, fp, seg_info)

        brief = generate_recommendation(
            event_spec, fp, plan, result, before, after,
            seg_info=seg_info, event_date=date_str,
        )
        print(brief)
