"""
============================================================================
STAGE 6 — DELTA PREDICTOR
============================================================================
Trains on historical event rows (already have a measured delta) and learns
to forecast the congestion footprint for a FUTURE event — before it happens.

The key spatial feature is hop_from_event: how many road-graph hops away
each segment is from the event junction. Closer = more impacted.

Training:  event_active rows from panel_with_deltas  (measured delta as label)
Inference: given (event_type, junction, crowd, timing) → predicted delta
           per segment per time-bin, with ± confidence interval
"""
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from config import HOP_DECAY, MAX_HOPS, MODELS_DIR, BASELINE_FEATURES, BIN_MINUTES
from impact import hop_distance_map

PREDICTOR_FEATURES = [
    "hop_from_event", "hop_weight",
    "capacity", "lanes", "free_flow_speed",
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "is_weekend", "is_rain",
    "crowd", "crowd_log",
    "event_type_code",
    "segment_code",
    "baseline_predicted",
]

EVENT_TYPE_MAP = {
    "public_event": 0, "procession": 1, "protest": 2,
    "congestion": 3, "vip_movement": 4, "construction": 5, "none": 6,
}

_WEEKDAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

LGBM_PREDICTOR_PARAMS = {
    "objective":         "regression",
    "n_estimators":      300,
    "max_depth":         5,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "min_child_samples": 5,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "random_state":      42,
    "verbose":           -1,
}


# ---------------------------------------------------------------------------
# Hop feature computation
# ---------------------------------------------------------------------------
def _compute_hop_features(G, seg_info, junction_node):
    """
    For a given event junction, return {segment_id: {hop_from_event, hop_weight}}
    for every segment within MAX_HOPS.
    """
    hop_map = hop_distance_map(G, junction_node)
    max_cap = max(s["capacity"] for s in seg_info.values()) if seg_info else 1.0
    result = {}
    for sid, info in seg_info.items():
        hop = min(hop_map.get(info["u"], 999), hop_map.get(info["v"], 999))
        hop = min(hop, MAX_HOPS + 1)
        base_w = HOP_DECAY.get(hop, 0.0)
        cap_ratio = info["capacity"] / max_cap
        weight = base_w * (1.0 - 0.4 * cap_ratio)
        result[sid] = {"hop_from_event": int(hop), "hop_weight": round(float(weight), 4)}
    return result


# ---------------------------------------------------------------------------
# Training data builder
# ---------------------------------------------------------------------------
def build_predictor_training_data(panel_d, event_days, G, seg_info):
    """
    Enrich event-active panel rows with hop/crowd features.
    Returns a DataFrame ready for DeltaPredictor.train().

    Only dates present in both panel_d (event_active rows) AND event_days
    (i.e. with a valid graph junction) contribute training rows.
    """
    ev = panel_d[panel_d["event_active"]].copy()
    if ev.empty:
        print("  [delta_predictor] No event-active rows — cannot build training data.")
        return pd.DataFrame()

    # 1. Pre-compute hop features per junction (one BFS per unique junction)
    jn_hop_cache = {}
    for date_str, meta in event_days.items():
        jn = meta["junction"]
        if jn not in jn_hop_cache and jn in G.nodes:
            jn_hop_cache[jn] = _compute_hop_features(G, seg_info, jn)

    # 2. Build a flat (date, segment_id) → hop features table
    hop_rows = []
    for date_str, meta in event_days.items():
        jn = meta["junction"]
        feats = jn_hop_cache.get(jn)
        if feats is None:
            continue
        for sid, hf in feats.items():
            hop_rows.append({
                "date":           date_str,
                "segment_id":     sid,
                "hop_from_event": hf["hop_from_event"],
                "hop_weight":     hf["hop_weight"],
            })
    if not hop_rows:
        print("  [delta_predictor] No valid event junctions in graph. Check event_days.")
        return pd.DataFrame()

    hop_df = pd.DataFrame(hop_rows)

    # 3. Event-level metadata table
    meta_df = pd.DataFrame([
        {
            "date":            d,
            "crowd":           int(meta["crowd"]),
            "crowd_log":       float(np.log1p(meta["crowd"])),
            "event_type_code": int(EVENT_TYPE_MAP.get(meta["type"], 6)),
        }
        for d, meta in event_days.items()
    ])

    # 4. Merge — vectorised, no iterrows
    result = (
        ev
        .merge(hop_df,  on=["date", "segment_id"], how="inner")
        .merge(meta_df, on="date",                 how="left")
    )

    # 5. Keep only rows within reach of the event
    result = result[result["hop_from_event"] <= MAX_HOPS].copy()
    print(f"  [delta_predictor] Training rows: {len(result):,} "
          f"({result['date'].nunique()} event-dates, "
          f"{result['segment_id'].nunique()} segments)")
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------
class DeltaPredictor:
    """Predicts per-segment congestion delta for a future (unseen) event."""

    def __init__(self):
        self.model = None
        self.metrics = {}
        self._type_std = {}       # per-event-type residual std for CI

    # ------------------------------------------------------------------
    def train(self, training_data):
        """
        Fit on enriched event rows.

        Uses a time-based split (first 80% of event-dates = train,
        last 20% = test) to simulate forecasting unseen future events.
        """
        df = training_data.dropna(subset=PREDICTOR_FEATURES + ["delta"]).copy()
        if df.empty:
            raise ValueError("Empty training data — nothing to train on.")

        X = df[PREDICTOR_FEATURES].astype(float)
        y = df["delta"]

        dates = sorted(df["date"].unique())
        cut_idx = max(1, int(len(dates) * 0.8))
        cut = dates[cut_idx - 1]

        train_m = df["date"] <= cut
        test_m  = df["date"] >  cut

        self.model = lgb.LGBMRegressor(**LGBM_PREDICTOR_PARAMS)
        self.model.fit(X[train_m], y[train_m])

        preds_train = self.model.predict(X[train_m])
        df.loc[train_m, "_resid"] = y[train_m].values - preds_train

        # Per-event-type std for uncertainty intervals
        for et, grp in df[train_m].groupby("event_type"):
            self._type_std[str(et)] = float(grp["_resid"].std())

        train_mae = float(mean_absolute_error(y[train_m], preds_train))
        test_mae  = None
        if test_m.sum() > 0:
            preds_test = self.model.predict(X[test_m])
            test_mae   = float(mean_absolute_error(y[test_m], preds_test))

        self.metrics = {
            "train_rows":       int(train_m.sum()),
            "test_rows":        int(test_m.sum()),
            "train_mae_kmh":    round(train_mae, 3),
            "test_mae_kmh":     round(test_mae, 3) if test_mae else None,
            "event_types_seen": list(df["event_type"].unique()),
            "cutoff_date":      cut,
        }
        return self.metrics

    # ------------------------------------------------------------------
    def predict_footprint(self, event_spec, G, seg_info, baseline_model,
                           seg_categories=None):
        """
        Predict the congestion footprint for a future event.

        Parameters
        ----------
        event_spec : dict
            type       – event type string
            junction   – OSMnx node ID of the event location
            crowd      – expected crowd size
            start_h    – event start hour (float, e.g. 19.0)
            end_h      – event end hour
            weekday    – day name (e.g. "Saturday")
            is_weekend – bool
            is_rain    – bool

        Returns
        -------
        DataFrame: segment_id, road, hour, time, hop_from_event,
                   capacity, baseline_predicted, delta_pred, delta_lo, delta_hi
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        jn = event_spec["junction"]
        if jn not in G.nodes:
            raise ValueError(f"Junction {jn!r} not found in road graph.")

        hop_feats  = _compute_hop_features(G, seg_info, jn)
        crowd      = int(event_spec["crowd"])
        et_code    = int(EVENT_TYPE_MAP.get(event_spec.get("type", "none"), 6))
        resid_std  = self._type_std.get(event_spec.get("type", "none"), 2.5)
        is_weekend = int(bool(event_spec.get("is_weekend", False)))
        is_rain    = int(bool(event_spec.get("is_rain", False)))

        wd = _WEEKDAYS.index(event_spec.get("weekday", "Saturday")) \
             if event_spec.get("weekday") in _WEEKDAYS else 5
        wd_sin = float(np.sin(2 * np.pi * wd / 7))
        wd_cos = float(np.cos(2 * np.pi * wd / 7))

        # Time window: [start_h − 1, end_h + 1], 15-min bins
        all_hours = np.arange(0, 24, BIN_MINUTES / 60)
        active_h  = all_hours[
            (all_hours >= event_spec["start_h"] - 1) &
            (all_hours <= event_spec["end_h"]   + 1)
        ]

        rows = []
        for sid, info in seg_info.items():
            hf = hop_feats.get(sid)
            if hf is None or hf["hop_from_event"] > MAX_HOPS:
                continue
            for h in active_h:
                rows.append({
                    "segment_id":      sid,
                    "road":            info["road"],
                    "hour":            float(h),
                    "time":            f"{int(h):02d}:{int((h % 1)*60):02d}",
                    "hop_from_event":  hf["hop_from_event"],
                    "hop_weight":      hf["hop_weight"],
                    "capacity":        float(info["capacity"]),
                    "lanes":           int(info["lanes"]),
                    "free_flow_speed": float(info["ffs"]),
                    "hour_sin":        float(np.sin(2 * np.pi * h / 24)),
                    "hour_cos":        float(np.cos(2 * np.pi * h / 24)),
                    "weekday_sin":     wd_sin,
                    "weekday_cos":     wd_cos,
                    "is_weekend":      is_weekend,
                    "is_rain":         is_rain,
                    "crowd":           crowd,
                    "crowd_log":       float(np.log1p(crowd)),
                    "event_type_code": et_code,
                })

        if not rows:
            return pd.DataFrame()

        feat_df = pd.DataFrame(rows)

        # Baseline prediction (normal speed counterfactual)
        bl_input = feat_df.copy()

        if seg_categories is not None:
            cat = pd.Categorical(bl_input["segment_id"], categories=seg_categories)
            seg_code = cat.codes.astype("int8")
        else:
            seg_code = pd.array([0] * len(bl_input), dtype="int8")

        bl_input["segment_code"] = seg_code
        feat_df["segment_code"]  = seg_code   # needed for PREDICTOR_FEATURES

        bl_input["is_rain"]    = bl_input["is_rain"].astype(bool)
        bl_input["is_weekend"] = bl_input["is_weekend"].astype(bool)

        try:
            feat_df["baseline_predicted"] = baseline_model.predict(bl_input)
        except Exception:
            feat_df["baseline_predicted"] = feat_df["free_flow_speed"] * 0.8

        # Delta prediction with confidence interval (±1.5σ ≈ 87% CI)
        X = feat_df[PREDICTOR_FEATURES].astype(float)
        feat_df["delta_pred"] = self.model.predict(X)
        feat_df["delta_lo"]   = feat_df["delta_pred"] - 1.5 * resid_std
        feat_df["delta_hi"]   = feat_df["delta_pred"] + 1.5 * resid_std

        out_cols = ["segment_id", "road", "hour", "time", "hop_from_event",
                    "capacity", "baseline_predicted",
                    "delta_pred", "delta_lo", "delta_hi"]
        return feat_df[out_cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    def feature_importance(self):
        if self.model is None:
            return {}
        return dict(sorted(
            zip(PREDICTOR_FEATURES, self.model.feature_importances_),
            key=lambda x: -x[1],
        ))

    def save(self, path=None):
        path = path or (MODELS_DIR / "delta_predictor.joblib")
        joblib.dump({
            "model":    self.model,
            "metrics":  self.metrics,
            "type_std": self._type_std,
        }, path)
        return path

    @classmethod
    def load(cls, path=None):
        path = path or (MODELS_DIR / "delta_predictor.joblib")
        blob = joblib.load(path)
        obj  = cls()
        obj.model     = blob["model"]
        obj.metrics   = blob["metrics"]
        obj._type_std = blob["type_std"]
        return obj


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from road_network import build_road_graph
    from data_loader import load_clean_events
    from events_calendar import derive_event_days
    from traffic_data import build_speed_panel
    from features import build_feature_table, split_baseline_data
    from baseline_model import BaselineModel
    from delta import compute_deltas

    print("Loading pipeline artifacts...")
    G, seg_info = build_road_graph()
    events      = load_clean_events(zone_only=True, G=G)
    event_days  = derive_event_days(events, zone_only=True)
    panel       = build_speed_panel(G, seg_info, event_days)
    featured, cats = build_feature_table(panel)
    split       = split_baseline_data(featured)
    bm          = BaselineModel()
    bm.train(split, cats)
    panel_d     = compute_deltas(featured, bm)

    print("\n[Stage 6] Building predictor training data...")
    train_df = build_predictor_training_data(panel_d, event_days, G, seg_info)

    print("\n[Stage 6] Training delta predictor...")
    dp = DeltaPredictor()
    m  = dp.train(train_df)
    print(f"  Train MAE: {m['train_mae_kmh']} km/h  |  Test MAE: {m['test_mae_kmh']} km/h")
    print(f"  Feature importances:")
    for feat, imp in dp.feature_importance().items():
        print(f"    {feat:<22} {imp}")
    dp.save()

    # Predict footprint for a future event using first calendar junction
    sample_date, sample_ev = next(iter(event_days.items()))
    jn = sample_ev["junction"]
    if jn in G.nodes:
        print(f"\n[Stage 6] Predicting footprint for future public_event at {jn}...")
        fp = dp.predict_footprint({
            "type":      "public_event",
            "junction":  jn,
            "crowd":     35000,
            "start_h":   19.0,
            "end_h":     23.0,
            "weekday":   "Saturday",
            "is_weekend": True,
            "is_rain":   False,
        }, G, seg_info, bm, cats)
        print(f"  {len(fp)} segment-time rows predicted")
        worst = fp.sort_values("delta_pred").head(5)
        for _, r in worst.iterrows():
            print(f"  {r['road']:<28} hop={r['hop_from_event']}  "
                  f"delta={r['delta_pred']:+.1f} km/h  "
                  f"CI=[{r['delta_lo']:+.1f}, {r['delta_hi']:+.1f}]")
