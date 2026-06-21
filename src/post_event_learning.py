"""
============================================================================
STAGE 11 — POST-EVENT LEARNING LOOP
============================================================================
After each real event the system:
  1. Predicts the footprint using the CURRENT model (before the event)
  2. Compares that prediction to the ACTUAL delta from the speed panel
  3. Computes RMSE / MAE for this event
  4. Adds the event's rows to the cumulative training set
  5. Retrains the delta predictor
  6. Saves the event as a reusable template

The key deliverable is a chronological RMSE curve that shows prediction
error shrinking as more real events are observed — the learning flywheel.

This module is self-contained: give it the pipeline artifacts and it
returns history + templates without touching the rest of the pipeline.
"""
import json
import copy
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import MODELS_DIR, OUTPUTS_DIR, MAX_HOPS, HOP_DECAY, BIN_MINUTES
from delta_predictor import (
    DeltaPredictor, build_predictor_training_data,
    PREDICTOR_FEATURES, LGBM_PREDICTOR_PARAMS, EVENT_TYPE_MAP,
    _compute_hop_features,
)

LEARNING_HISTORY_PATH = OUTPUTS_DIR / "learning_history.json"
TEMPLATES_PATH        = OUTPUTS_DIR / "event_templates.json"
RETRAINED_MODEL_PATH  = MODELS_DIR  / "delta_predictor_retrained.joblib"


# ---------------------------------------------------------------------------
# Per-event error measurement
# ---------------------------------------------------------------------------
def evaluate_event_prediction(event_date, footprint_df, panel_d):
    """
    Compare delta_pred (footprint) to actual delta (panel_d) for one event.

    Returns a dict:
      n_matched  – rows matched on (segment_id, hour)
      mae        – mean absolute error (km/h)
      rmse       – root mean squared error (km/h)
      bias       – mean(predicted − actual) — positive = over-predicted severity
      worst_pred – worst predicted delta
      worst_actual – worst actual delta
    """
    actual = panel_d[
        (panel_d["date"] == event_date) & panel_d["event_active"]
    ][["segment_id", "hour", "delta"]].copy()

    if actual.empty or footprint_df.empty:
        return None

    merged = footprint_df.merge(actual, on=["segment_id", "hour"], how="inner")
    if merged.empty:
        return None

    errors = merged["delta_pred"] - merged["delta"]
    return {
        "n_matched":    int(len(merged)),
        "mae":          float(round(mean_absolute_error(merged["delta"], merged["delta_pred"]), 3)),
        "rmse":         float(round(mean_squared_error(merged["delta"], merged["delta_pred"]) ** 0.5, 3)),
        "bias":         float(round(errors.mean(), 3)),
        "worst_pred":   float(round(merged["delta_pred"].min(), 2)),
        "worst_actual": float(round(merged["delta"].min(), 2)),
        "r2":           float(round(
            1 - ((merged["delta"] - merged["delta_pred"]) ** 2).sum()
            / max(((merged["delta"] - merged["delta"].mean()) ** 2).sum(), 1e-9),
            3
        )),
    }


# ---------------------------------------------------------------------------
# Incremental retraining
# ---------------------------------------------------------------------------
def retrain_on_accumulated(cumulative_train_df, prev_type_std=None):
    """
    Fit a fresh LightGBM model on cumulative_train_df.
    Returns (model, type_std_dict, metrics_dict).
    """
    df = cumulative_train_df.dropna(subset=PREDICTOR_FEATURES + ["delta"]).copy()
    if len(df) < 20:
        return None, prev_type_std or {}, {}

    X = df[PREDICTOR_FEATURES].astype(float)
    y = df["delta"]

    model = lgb.LGBMRegressor(**LGBM_PREDICTOR_PARAMS)
    model.fit(X, y)

    preds  = model.predict(X)
    resids = y.values - preds

    type_std = dict(prev_type_std or {})
    for et, grp in df.groupby("event_type"):
        mask = df["event_type"] == et
        type_std[str(et)] = float(np.std(resids[mask.values]))

    mae  = float(round(mean_absolute_error(y, preds), 3))
    rmse = float(round(mean_squared_error(y, preds) ** 0.5, 3))

    return model, type_std, {
        "train_rows":    len(df),
        "train_mae_kmh": mae,
        "train_rmse_kmh": rmse,
    }


# ---------------------------------------------------------------------------
# Event template builder
# ---------------------------------------------------------------------------
def build_event_template(event_date, event_spec, actual_error_metrics,
                          footprint_df, panel_d):
    """
    Save the fingerprint of a completed event as a template.

    A template captures the proven impact pattern so future similar events
    can use it as a prior — the warm-start for new event types or venues.
    """
    actual = panel_d[
        (panel_d["date"] == event_date) & panel_d["event_active"]
    ][["segment_id", "delta", "hour"]]

    template = {
        "event_date":    event_date,
        "event_type":    event_spec.get("type", "unknown"),
        "crowd":         int(event_spec.get("crowd", 0)),
        "crowd_source":  event_spec.get("crowd_source", "prior"),
        "start_h":       float(event_spec.get("start_h", 0)),
        "end_h":         float(event_spec.get("end_h", 0)),
        "prediction_error": actual_error_metrics or {},
    }

    if not actual.empty:
        template["actual_impact"] = {
            "avg_delta_kmh":  float(round(actual["delta"].mean(), 2)),
            "worst_delta_kmh": float(round(actual["delta"].min(), 2)),
            "severe_segs":    int((actual["delta"] < -10).sum()),
            "moderate_segs":  int(((actual["delta"] >= -10) & (actual["delta"] < -5)).sum()),
            "peak_hour":      float(actual.groupby("hour")["delta"].mean().idxmin()),
        }

    if footprint_df is not None and not footprint_df.empty:
        template["predicted_impact"] = {
            "avg_delta_pred":   float(round(footprint_df["delta_pred"].mean(), 2)),
            "worst_delta_pred": float(round(footprint_df["delta_pred"].min(), 2)),
        }

    return template


# ---------------------------------------------------------------------------
# Main learning loop
# ---------------------------------------------------------------------------
def run_historical_loop(event_days, panel_d, G, seg_info, bm, seg_cats,
                         seed_n_events=3, verbose=True):
    """
    Chronological leave-one-out learning simulation.

    Walk through event_days in date order. For the first seed_n_events, train
    from scratch (cold start). For each subsequent event:
      - Predict with the model trained on all prior events
      - Measure RMSE against actual delta in panel_d
      - Add event to cumulative training set
      - Retrain
      - Save template

    Returns
    -------
    learning_history : list of dicts (one per event)
    event_templates  : dict {date: template}
    final_predictor  : DeltaPredictor trained on all events
    """
    sorted_dates = sorted(event_days.keys())
    if not sorted_dates:
        return [], {}, None

    # Build full training data once (all events), then slice chronologically
    all_train_df = build_predictor_training_data(panel_d, event_days, G, seg_info)
    if all_train_df.empty:
        if verbose:
            print("  [learning] No training data — nothing to learn from.")
        return [], {}, None

    # Map date → rows in all_train_df for efficient slicing
    date_groups = {d: grp for d, grp in all_train_df.groupby("date")}

    cumulative_rows = []        # grows each iteration
    current_model   = None
    current_type_std = {}
    learning_history = []
    event_templates  = {}
    baseline_rmse    = None

    for i, event_date in enumerate(sorted_dates):
        ev = event_days[event_date]
        jn = ev["junction"]
        if jn not in G.nodes:
            continue

        event_spec = {**ev, "weekday": "Saturday", "is_weekend": True, "is_rain": False}

        # --- Predict with current model (before training on this event) ---
        pred_error = None
        footprint  = None

        if current_model is not None and i >= seed_n_events:
            dp_tmp         = DeltaPredictor()
            dp_tmp.model   = current_model
            dp_tmp._type_std = current_type_std
            try:
                footprint = dp_tmp.predict_footprint(
                    event_spec, G, seg_info, bm, seg_cats
                )
                pred_error = evaluate_event_prediction(event_date, footprint, panel_d)
            except Exception as e:
                if verbose:
                    print(f"  [learning] Prediction error for {event_date}: {e}")

        # --- Accumulate this event's training rows ---
        new_rows = date_groups.get(event_date)
        if new_rows is not None and len(new_rows) > 0:
            cumulative_rows.append(new_rows)

        if not cumulative_rows:
            continue

        cumulative_df = pd.concat(cumulative_rows, ignore_index=True)

        # --- Retrain ---
        new_model, new_type_std, train_metrics = retrain_on_accumulated(
            cumulative_df, prev_type_std=current_type_std
        )
        if new_model is not None:
            current_model    = new_model
            current_type_std = new_type_std

        # --- Record history ---
        rmse = pred_error["rmse"] if pred_error else None
        if rmse is not None and baseline_rmse is None:
            baseline_rmse = rmse

        improvement_pct = None
        if rmse is not None and baseline_rmse and baseline_rmse > 0:
            improvement_pct = round(100 * (baseline_rmse - rmse) / baseline_rmse, 1)

        entry = {
            "event_n":         i + 1,
            "date":            event_date,
            "event_type":      ev["type"],
            "crowd":           int(ev["crowd"]),
            "crowd_source":    ev.get("crowd_source", "prior"),
            "cumulative_events": len(cumulative_rows),
            "train_rows":      train_metrics.get("train_rows", 0),
            "train_mae_kmh":   train_metrics.get("train_mae_kmh"),
            "train_rmse_kmh":  train_metrics.get("train_rmse_kmh"),
            "pred_mae":        pred_error["mae"]  if pred_error else None,
            "pred_rmse":       pred_error["rmse"] if pred_error else None,
            "pred_bias":       pred_error["bias"] if pred_error else None,
            "pred_n_matched":  pred_error["n_matched"] if pred_error else None,
            "improvement_pct": improvement_pct,
        }
        learning_history.append(entry)

        if verbose:
            rmse_str = f"RMSE={rmse:.2f}" if rmse else "(seed — no prediction yet)"
            print(f"  [learning] Event {i+1:2d}: {event_date}  {ev['type']:<20} "
                  f"{rmse_str}  train_rows={train_metrics.get('train_rows',0)}")

        # --- Save event template ---
        template = build_event_template(event_date, event_spec, pred_error,
                                        footprint, panel_d)
        event_templates[event_date] = template

    # --- Wrap up: build the final DeltaPredictor on all events ---
    final_dp = None
    if current_model is not None:
        final_dp           = DeltaPredictor()
        final_dp.model     = current_model
        final_dp._type_std = current_type_std
        final_dp.metrics   = {
            "train_rows":    len(cumulative_df) if cumulative_rows else 0,
            "event_types_seen": list(all_train_df["event_type"].unique()),
        }

    return learning_history, event_templates, final_dp


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def save_learning_artifacts(learning_history, event_templates, final_dp=None):
    """Write history and templates to outputs/; optionally save retrained model."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(LEARNING_HISTORY_PATH, "w") as f:
        json.dump(learning_history, f, indent=2, default=str)

    with open(TEMPLATES_PATH, "w") as f:
        json.dump(event_templates, f, indent=2, default=str)

    if final_dp is not None:
        final_dp.save(RETRAINED_MODEL_PATH)

    return LEARNING_HISTORY_PATH, TEMPLATES_PATH


def load_learning_artifacts():
    """Load previously saved history and templates. Returns (history, templates)."""
    history   = json.load(open(LEARNING_HISTORY_PATH)) if LEARNING_HISTORY_PATH.exists() else []
    templates = json.load(open(TEMPLATES_PATH))         if TEMPLATES_PATH.exists()        else {}
    return history, templates


# ---------------------------------------------------------------------------
# Summarise learning results for display
# ---------------------------------------------------------------------------
def learning_summary(learning_history):
    """Return a human-readable summary string for the pipeline output."""
    measured = [e for e in learning_history if e.get("pred_rmse") is not None]
    if not measured:
        return "  No prediction measurements (all events used as seed data)."

    first_rmse = measured[0]["pred_rmse"]
    last_rmse  = measured[-1]["pred_rmse"]
    best_rmse  = min(e["pred_rmse"] for e in measured)
    improvement = 100 * (first_rmse - best_rmse) / first_rmse if first_rmse > 0 else 0

    lines = [
        f"  Events measured  : {len(measured)}",
        f"  First-event RMSE : {first_rmse:.2f} km/h",
        f"  Last-event  RMSE : {last_rmse:.2f} km/h",
        f"  Best RMSE        : {best_rmse:.2f} km/h",
        f"  Improvement      : {improvement:.1f}%  ({first_rmse:.2f} → {best_rmse:.2f} km/h)",
    ]
    return "\n".join(lines)
