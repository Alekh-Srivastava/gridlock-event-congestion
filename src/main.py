"""
============================================================================
MAIN — run the full pipeline end to end
============================================================================
    python main.py

Stages:
  0. Build road graph + compute centrality
  1. Load & clean ASTraM events  (data_loader)
  2. Build speed panel           (traffic_data — swap for real TomTom)
  3. Engineer features           (features)
  4. Train baseline model        (baseline_model)
  5. Compute deltas + validate   (delta)
  6. Per-event impact reports    (delta output layers)
  7. Train delta predictor       (delta_predictor)  ← Stage 6
  8. Simulate interventions      (simulation)       ← Stage 7
  9. Optimize deployment         (optimizer)        ← Stage 8
  10. Generate deployment brief  (recommender)      ← Stage 9
  11. Persist models & outputs
"""
import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd

from config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR, VENUE_NAME
from road_network import build_road_graph
from data_loader import load_clean_events
from events_calendar import derive_event_days, summarize_calendar
from traffic_data import build_speed_panel
from features import build_feature_table, split_baseline_data
from baseline_model import BaselineModel
from impact import segment_centrality
from delta import (compute_deltas, validate_baseline,
                   priority_ranking, aggregate_impact)
from delta_predictor import DeltaPredictor, build_predictor_training_data
from simulation import EventSimulator
from optimizer import generate_candidates, InterventionOptimizer
from recommender import generate_recommendation
from post_event_learning import (
    run_historical_loop, save_learning_artifacts, learning_summary
)

# Toggle: True = synthetic speeds (sandbox), False = real TomTom (needs key)
USE_SYNTHETIC = True
TOMTOM_KEY = None  # set your TomTom API key here to use real speeds


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def main():
    banner(f"EVENT-DRIVEN CONGESTION SYSTEM  —  {VENUE_NAME}")

    # --- Stage 0: graph -----------------------------------------------------
    print("\n[0] Building road graph...")
    G, seg_info = build_road_graph()
    seg_bc = segment_centrality(G, seg_info)
    print(f"    {G.number_of_nodes()} junctions, {G.number_of_edges()} segments")

    # --- Stage 1: events ----------------------------------------------------
    print("\n[1] Loading & cleaning ASTraM events...")
    events = load_clean_events(zone_only=True, G=G)
    events.to_parquet(PROCESSED_DATA_DIR / "clean_events.parquet")

    # --- Derive REAL event calendar from the CSV (not hard-coded) -----------
    print("\n[1b] Deriving event calendar from real ASTraM data...")
    event_days = derive_event_days(events, zone_only=True)
    print(f"    {len(event_days)} real event-days derived")
    by_type = {}
    for ev in event_days.values():
        by_type[ev["type"]] = by_type.get(ev["type"], 0) + 1
    print(f"    By type: {by_type}")

    # --- Stage 2: traffic speed panel --------------------------------------
    print("\n[2] Building speed panel (Plane B)...")
    panel = build_speed_panel(G, seg_info, event_days,
                              use_synthetic=USE_SYNTHETIC, tomtom_key=TOMTOM_KEY)

    # --- Stage 3: features --------------------------------------------------
    print("\n[3] Engineering features...")
    featured, seg_cats = build_feature_table(panel)
    split = split_baseline_data(featured)
    print(f"    Train: {len(split['X_train']):,} rows (non-event, before {split['cutoff'].date()})")
    print(f"    Test:  {len(split['X_test']):,} rows")
    print(f"    Excluded {split['n_excluded_event_rows']:,} event-day rows from baseline")

    # --- Stage 4: baseline model -------------------------------------------
    print("\n[4] Training baseline model...")
    bm = BaselineModel()
    metrics = bm.train(split, seg_cats)
    print(f"    Test MAE:  {metrics['test_mae']:.2f} km/h")
    print(f"    Test RMSE: {metrics['test_rmse']:.2f} km/h")
    bm.save()

    # --- Stage 5: deltas + validation --------------------------------------
    print("\n[5] Computing deltas...")
    panel_d = compute_deltas(featured, bm)
    checks = validate_baseline(panel_d)
    print(f"    Non-event delta mean: {checks['non_event_delta_mean']:+.3f} km/h (want ~0)")
    print(f"    Noise floor (std):    {checks['non_event_delta_std']:.2f} km/h")
    panel_d.to_parquet(PROCESSED_DATA_DIR / "panel_with_deltas.parquet")

    # --- Stage 6: per-event impact reports ---------------------------------
    banner("EVENT IMPACT REPORTS")
    all_reports = []
    # report on the highest-crowd event days (most interesting)
    ranked_days = sorted(event_days.items(),
                         key=lambda kv: kv[1]["crowd"], reverse=True)[:8]
    for date_str, ev in ranked_days:
        agg = aggregate_impact(panel_d, date_str)
        if not agg:
            continue
        rank = priority_ranking(panel_d, date_str, seg_bc, seg_info)
        print(f"\n  {date_str}  ({ev['type']}, ~{ev['crowd']:,} crowd [{ev['crowd_source']}])")
        print(f"    Congestion increase : {agg['congestion_increase_pct']:.0f}%")
        print(f"    Worst delta         : {agg['worst_delta_kmh']:.1f} km/h")
        print(f"    Severe segments     : {agg['severe_segments']}")
        if not rank.empty:
            top = rank.iloc[0]
            print(f"    #1 target           : {top['road']} (Δ {top['worst_delta']:.1f})")
        agg["top_targets"] = rank.head(3)["road"].tolist() if not rank.empty else []
        all_reports.append(agg)

    # Persist legacy impact reports
    with open(OUTPUTS_DIR / "impact_reports.json", "w") as f:
        json.dump(all_reports, f, indent=2, default=str)

    # =========================================================================
    # Stage 7 — DELTA PREDICTOR (forecast future events)
    # =========================================================================
    banner("STAGE 7 — DELTA PREDICTOR (planning mode)")
    print("\n[7] Building predictor training data from event-active rows...")
    train_df = build_predictor_training_data(panel_d, event_days, G, seg_info)

    if train_df.empty:
        print("  WARNING: no training data for delta predictor — skipping Stages 7-10.")
        banner("PIPELINE COMPLETE (Stages 0-6)")
        return

    print("\n[7] Training delta predictor...")
    dp = DeltaPredictor()
    dp_metrics = dp.train(train_df)
    print(f"    Train MAE : {dp_metrics['train_mae_kmh']} km/h")
    print(f"    Test  MAE : {dp_metrics['test_mae_kmh']} km/h")
    print(f"    Event types seen: {dp_metrics['event_types_seen']}")
    dp.save()

    # =========================================================================
    # Stages 8-10 — run for the highest-crowd event in the calendar
    # =========================================================================
    crowd_events = {d: ev for d, ev in event_days.items() if ev.get("crowd", 0) > 0}
    if not crowd_events:
        print("  No crowd events found — skipping simulation / optimizer.")
        banner("PIPELINE COMPLETE (Stages 0-7)")
        return

    # Prefer daytime events (start_h >= 8) so the deployment brief is sensible;
    # fall back to highest-crowd event if none qualify.
    daytime = {d: ev for d, ev in crowd_events.items() if ev.get("start_h", 0) >= 8}
    pool     = daytime if daytime else crowd_events
    demo_date, demo_ev = max(pool.items(), key=lambda kv: kv[1]["crowd"])
    demo_jn = demo_ev["junction"]

    if demo_jn not in G.nodes:
        print(f"  Demo junction {demo_jn!r} not in graph — skipping Stages 8-10.")
        banner("PIPELINE COMPLETE (Stages 0-7)")
        return

    demo_spec = {
        "type":       demo_ev["type"],
        "junction":   demo_jn,
        "crowd":      demo_ev["crowd"],
        "crowd_source": demo_ev.get("crowd_source", "prior"),
        "start_h":    demo_ev["start_h"],
        "end_h":      demo_ev["end_h"],
        "weekday":    "Saturday",
        "is_weekend": True,
        "is_rain":    False,
    }

    banner(f"STAGE 8 — FOOTPRINT PREDICTION  ({demo_date}, crowd {demo_ev['crowd']:,})")
    print("\n[8] Predicting congestion footprint for this event...")
    footprint = dp.predict_footprint(demo_spec, G, seg_info, bm, seg_cats)
    print(f"    {len(footprint)} segment-time rows predicted")
    if not footprint.empty:
        worst = footprint.sort_values("delta_pred").head(5)
        print("    Top 5 worst-predicted segments:")
        for _, r in worst.iterrows():
            print(f"      {r['road']:<30} hop={r['hop_from_event']}  "
                  f"delta={r['delta_pred']:+.1f} km/h  "
                  f"[{r['delta_lo']:+.1f}, {r['delta_hi']:+.1f}]")

    # =========================================================================
    # Stage 9 — SIMULATION
    # =========================================================================
    banner("STAGE 9 — SIMULATION (BPR intervention measurement)")
    print("\n[9] Generating intervention candidates...")
    candidates = generate_candidates(footprint, seg_bc, seg_info, G)
    print(f"    {len(candidates)} candidates: "
          f"{sum(1 for c in candidates if c['type']=='officer')} officers, "
          f"{sum(1 for c in candidates if c['type']=='barricade')} barricades, "
          f"{sum(1 for c in candidates if c['type']=='diversion')} diversions")

    sim = EventSimulator(G, seg_info)
    print("\n[9] Evaluating individual intervention effects...")
    iv_df, base_speeds = sim.measure_interventions(demo_spec, footprint, candidates[:12])
    if not iv_df.empty:
        print("    Top individual interventions:")
        for _, r in iv_df.head(5).iterrows():
            print(f"      [{r['type']:9s}] {str(r['label']):<45} "
                  f"+{r['congestion_removed_kmh']:.2f} km/h "
                  f"({r['segments_improved']} segs)")

    # =========================================================================
    # Stage 10 — OPTIMIZER
    # =========================================================================
    banner("STAGE 10 — OPTIMIZER (greedy maximal coverage)")
    print("\n[10] Finding optimal deployment under budget (4 officers, 2 barricades, 1 diversion)...")
    opt = InterventionOptimizer(sim)
    plan, summary, sim_before, sim_after = opt.optimize(
        demo_spec, footprint, candidates,
        n_officers=4, n_barricades=2, n_diversions=1,
    )
    opt_result = opt.report(plan, summary, sim_before, sim_after, footprint, seg_info)

    # =========================================================================
    # Stage 11 — RECOMMENDATION
    # =========================================================================
    banner("STAGE 11 — DEPLOYMENT BRIEF")
    brief = generate_recommendation(
        demo_spec, footprint, plan, opt_result,
        sim_before, sim_after,
        seg_info=seg_info, event_date=demo_date,
    )
    print(brief)

    # =========================================================================
    # Stage 11 — POST-EVENT LEARNING LOOP
    # =========================================================================
    banner("STAGE 11 — POST-EVENT LEARNING LOOP")
    print("\n[11] Running chronological learning loop over all historical events...")
    learning_history, event_templates, retrained_dp = run_historical_loop(
        event_days, panel_d, G, seg_info, bm, seg_cats,
        seed_n_events=3, verbose=True,
    )
    print(learning_summary(learning_history))
    save_learning_artifacts(learning_history, event_templates, retrained_dp)
    print(f"    Saved learning history ({len(learning_history)} events) → {OUTPUTS_DIR}/learning_history.json")
    print(f"    Saved event templates  ({len(event_templates)} events) → {OUTPUTS_DIR}/event_templates.json")

    # Save all outputs
    print(f"\n[12] Saving all outputs...")
    full_output = {
        "demo_event":   demo_date,
        "event_spec":   {k: v for k, v in demo_spec.items() if k != "junction"},
        "predictor_metrics": dp_metrics,
        "optimizer_result":  opt_result,
        "deployment_brief":  brief,
        "learning_history":  learning_history,
    }
    with open(OUTPUTS_DIR / "full_pipeline_output.json", "w") as f:
        json.dump(full_output, f, indent=2, default=str)
    print(f"    Saved to {OUTPUTS_DIR}/full_pipeline_output.json")
    print(f"    Models  → {MODELS_DIR}/")

    banner("PIPELINE COMPLETE — All 12 Stages")


if __name__ == "__main__":
    main()
