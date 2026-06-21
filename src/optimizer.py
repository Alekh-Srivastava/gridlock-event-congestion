"""
============================================================================
STAGE 8 — INTERVENTION OPTIMIZER
============================================================================
Given a budget (N officers, M barricades, K diversions), find the deployment
plan that maximises total simulated congestion reduction.

Algorithm: Greedy Maximal Coverage
  At each step, pick the next intervention with the highest MARGINAL gain
  (i.e. how much it adds ON TOP of already-selected ones, not in isolation).
  Greedy achieves ≥ (1 − 1/e) ≈ 63% of the global optimum for submodular
  coverage objectives — provably near-optimal without exhaustive search.

Search space pruning:
  We only consider segments/junctions that appear in the top-priority
  footprint (high |delta| × high centrality). This keeps the candidate
  set small (~20-30) and every candidate is simulation-validated.

Outputs:
  - Ranked deployment plan (which junction/segment, which resource type)
  - Before/after simulation comparison
  - Marginal contribution of each intervention
"""
import numpy as np
import pandas as pd

from config import MAX_HOPS
from impact import segment_impact_weights


# ---------------------------------------------------------------------------
# Candidate generator
# ---------------------------------------------------------------------------
def generate_candidates(footprint_df, seg_bc, seg_info, G,
                         n_officer_candidates=10,
                         n_barricade_candidates=5,
                         n_diversion_candidates=5):
    """
    Build the shortlist of interventions worth simulating.

    Officer candidates  → high-centrality junctions near impacted segments
    Barricade candidates → segments with worst delta AND high load (low capacity)
    Diversion candidates → high-delta segments with viable alternate routes

    Returns a list of dicts: {type, junction/segment_id, label, priority_score}
    """
    if footprint_df.empty:
        return []

    # Aggregate footprint to segment level
    seg_agg = (
        footprint_df
        .groupby("segment_id")
        .agg(
            worst_delta=("delta_pred", "min"),
            avg_delta=("delta_pred", "mean"),
            road=("road", "first"),
            capacity=("capacity", "first"),
            hop=("hop_from_event", "min"),
        )
        .reset_index()
    )
    seg_agg["centrality"]     = seg_agg["segment_id"].map(seg_bc).fillna(0.0)
    seg_agg["priority_score"] = (
        seg_agg["worst_delta"].abs() * (1 + seg_agg["centrality"] * 5)
    )
    seg_agg = seg_agg.sort_values("priority_score", ascending=False)

    candidates = []

    # Build a road-name lookup for junctions: junction → set of road names
    jn_roads = {}
    for sid, info in seg_info.items():
        road = str(info.get("road", "unnamed"))
        if road == "unnamed":
            continue
        for node in [info.get("u"), info.get("v")]:
            if node is not None:
                jn_roads.setdefault(node, set()).add(road)

    def _junction_label(jn):
        roads = jn_roads.get(jn, set())
        if roads:
            return " / ".join(sorted(roads)[:2]) + " Junction"
        return f"Junction {str(jn)[:12]}"

    # 1. OFFICER candidates: junctions of top-priority segments
    junction_scores = {}
    for _, row in seg_agg.iterrows():
        sid  = row["segment_id"]
        info = seg_info.get(sid, {})
        for node in [info.get("u"), info.get("v")]:
            if node is None or node not in G.nodes:
                continue
            junction_scores[node] = max(
                junction_scores.get(node, 0.0), float(row["priority_score"])
            )

    top_junctions = sorted(junction_scores, key=junction_scores.get, reverse=True)
    for jn in top_junctions[:n_officer_candidates]:
        candidates.append({
            "type":           "officer",
            "junction":       jn,
            "segment_id":     None,
            "label":          f"Officer @ {_junction_label(jn)}",
            "priority_score": junction_scores[jn],
        })

    # 2. BARRICADE candidates: worst-delta, low-capacity segments
    barricade_pool = seg_agg[
        (seg_agg["worst_delta"] < -3.0) &
        (seg_agg["capacity"] < seg_agg["capacity"].median())
    ].head(n_barricade_candidates)
    for _, row in barricade_pool.iterrows():
        candidates.append({
            "type":           "barricade",
            "junction":       None,
            "segment_id":     row["segment_id"],
            "label":          f"Barricade on {row['road']} ({row['segment_id'][:12]}…)",
            "priority_score": float(row["priority_score"]) * 0.8,
        })

    # 3. DIVERSION candidates: high-delta segments with multiple neighbors
    diversion_pool = seg_agg[seg_agg["worst_delta"] < -4.0].head(n_diversion_candidates)
    for _, row in diversion_pool.iterrows():
        sid  = row["segment_id"]
        info = seg_info.get(sid, {})
        u, v = info.get("u"), info.get("v")
        if u and v and G.number_of_edges(u, v) > 0:
            # Only suggest diversion if there is an alternate path
            try:
                import networkx as nx
                G_temp = G.copy()
                if G_temp.has_edge(u, v):
                    G_temp.remove_edge(u, v)
                nx.shortest_path(G_temp, u, v)
                candidates.append({
                    "type":           "diversion",
                    "junction":       None,
                    "segment_id":     sid,
                    "label":          f"Diversion away from {row['road']}",
                    "priority_score": float(row["priority_score"]) * 0.6,
                })
            except Exception:
                pass

    return sorted(candidates, key=lambda c: -c["priority_score"])


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
class InterventionOptimizer:
    """
    Greedy submodular coverage optimizer.
    Selects the best combination of officers/barricades/diversions under
    resource budget constraints, guided by simulation-measured gains.
    """

    def __init__(self, simulator):
        self.sim = simulator

    def optimize(self, event_spec, footprint_df, candidates,
                  n_officers=4, n_barricades=2, n_diversions=1):
        """
        Greedy selection: at each step pick the candidate with the highest
        marginal gain GIVEN the already-selected set. Simulate the combined
        effect rather than summing individual effects.

        Returns
        -------
        plan : list of selected intervention dicts
        summary : DataFrame with per-step marginal improvement
        sim_before : {segment_id: speed_kmh}  (no intervention)
        sim_after  : {segment_id: speed_kmh}  (all selected interventions)
        """
        budget = {
            "officer":   n_officers,
            "barricade": n_barricades,
            "diversion": n_diversions,
        }
        remaining = dict(budget)
        selected  = []
        steps     = []

        sim_before, _ = self.sim.measure_interventions(event_spec, footprint_df, [])
        # sim_before is the base_speeds dict from measure_interventions
        # We need the actual no-intervention speeds
        base_speeds = self.sim.run_no_intervention(event_spec, footprint_df)
        affected    = set(footprint_df["segment_id"].unique())

        def _avg_speed(speed_dict):
            vals = [speed_dict.get(s, 0) for s in affected]
            return float(np.mean(vals)) if vals else 0.0

        current_avg = _avg_speed(base_speeds)

        for step in range(n_officers + n_barricades + n_diversions):
            best_gain = -np.inf
            best_cand = None
            best_speeds = None

            for cand in candidates:
                if cand in selected:
                    continue
                if remaining.get(cand["type"], 0) <= 0:
                    continue

                trial = selected + [cand]
                new_speeds = self.sim.run_combined(event_spec, footprint_df, trial)
                new_avg    = _avg_speed(new_speeds)
                gain       = new_avg - current_avg

                if gain > best_gain:
                    best_gain   = gain
                    best_cand   = cand
                    best_speeds = new_speeds

            if best_cand is None or best_gain <= 0.01:
                break   # no more useful interventions

            selected.append(best_cand)
            remaining[best_cand["type"]] -= 1
            current_avg = _avg_speed(best_speeds)

            steps.append({
                "step":              step + 1,
                "type":              best_cand["type"],
                "label":             best_cand["label"],
                "marginal_gain_kmh": round(best_gain, 2),
                "cumulative_avg_kmh": round(current_avg, 2),
            })

        sim_after = self.sim.run_combined(event_spec, footprint_df, selected)

        summary = pd.DataFrame(steps)
        return selected, summary, base_speeds, sim_after

    def report(self, plan, summary, base_speeds, sim_after,
                footprint_df, seg_info):
        """Print a concise optimisation report."""
        affected = set(footprint_df["segment_id"].unique())

        before_avg = np.mean([base_speeds.get(s, 0) for s in affected])
        after_avg  = np.mean([sim_after.get(s, 0)   for s in affected])

        print(f"\n  Officers/resources selected: {len(plan)}")
        for iv in plan:
            print(f"    [{iv['type']:9s}] {iv['label']}")

        print(f"\n  Average corridor speed:")
        print(f"    Before interventions : {before_avg:.1f} km/h")
        print(f"    After  interventions : {after_avg:.1f} km/h")
        print(f"    Net improvement      : +{after_avg - before_avg:.1f} km/h")

        if not summary.empty:
            print(f"\n  Marginal contribution per step:")
            for _, r in summary.iterrows():
                print(f"    Step {int(r['step'])}: {r['label']:<45} "
                      f"+{r['marginal_gain_kmh']:.2f} km/h")

        return {
            "avg_speed_before": round(float(before_avg), 2),
            "avg_speed_after":  round(float(after_avg), 2),
            "net_improvement_kmh": round(float(after_avg - before_avg), 2),
            "plan": [{"type": iv["type"], "label": iv["label"]} for iv in plan],
        }


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

    G, seg_info  = build_road_graph()
    events       = load_clean_events(zone_only=True, G=G)
    event_days   = derive_event_days(events, zone_only=True)
    panel        = build_speed_panel(G, seg_info, event_days)
    featured, cats = build_feature_table(panel)
    split        = split_baseline_data(featured)
    bm           = BaselineModel(); bm.train(split, cats)
    panel_d      = compute_deltas(featured, bm)
    seg_bc       = segment_centrality(G, seg_info)

    train_df = build_predictor_training_data(panel_d, event_days, G, seg_info)
    dp = DeltaPredictor(); dp.train(train_df)

    crowd_events = {d: ev for d, ev in event_days.items() if ev["crowd"] > 0}
    date_str, ev = next(iter(crowd_events.items()))
    jn = ev["junction"]

    if jn not in G.nodes:
        print("Junction not in graph — skipping optimizer test.")
    else:
        event_spec = {
            "type": ev["type"], "junction": jn, "crowd": ev["crowd"],
            "start_h": ev["start_h"], "end_h": ev["end_h"],
            "weekday": "Saturday", "is_weekend": True, "is_rain": False,
        }
        fp = dp.predict_footprint(event_spec, G, seg_info, bm, cats)
        candidates = generate_candidates(fp, seg_bc, seg_info, G)
        print(f"\n[Stage 8] {len(candidates)} candidates generated")

        sim  = EventSimulator(G, seg_info)
        opt  = InterventionOptimizer(sim)
        plan, summary, before, after = opt.optimize(
            event_spec, fp, candidates,
            n_officers=4, n_barricades=2, n_diversions=1,
        )
        result = opt.report(plan, summary, before, after, fp, seg_info)
        print(f"\n  Optimizer result: {result}")
