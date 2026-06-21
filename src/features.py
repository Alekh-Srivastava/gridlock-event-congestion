"""
============================================================================
FEATURES — turn the raw speed panel into a model-ready dataset
============================================================================
This is the bridge between raw data and the baseline model. It adds the
engineered features the model needs and returns clean X / y splits.

Key design choices, explained:
  - HOUR and WEEKDAY are CYCLICALLY encoded (sin/cos) so the model knows
    23:00 is adjacent to 00:00, and Sunday is adjacent to Monday. A raw
    integer hour would wrongly treat 23 and 0 as far apart.
  - SEGMENT is label-encoded into segment_code. This is the feature that
    captures each road's demand/context (see the scalability note in README).
  - We persist the segment encoding so train and inference use identical codes.
"""
import numpy as np
import pandas as pd

from config import BASELINE_FEATURES, BASELINE_TARGET, TRAIN_TEST_SPLIT_QUANTILE


# Fixed weekday ordering so encoding is stable across runs
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]


def add_cyclical_time_features(df):
    """Add sin/cos encodings of hour and weekday."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    wd_num = df["weekday"].map({d: i for i, d in enumerate(_WEEKDAYS)})
    df["weekday_sin"] = np.sin(2 * np.pi * wd_num / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * wd_num / 7)
    return df


def encode_segments(df, categories=None):
    """
    Label-encode segment_id into segment_code.

    If `categories` is given (from training), reuse it so inference codes match.
    Returns (df, categories).
    """
    df = df.copy()
    if categories is None:
        cat = pd.Categorical(df["segment_id"])
        categories = cat.categories
    else:
        cat = pd.Categorical(df["segment_id"], categories=categories)
    df["segment_code"] = cat.codes
    return df, categories


def build_feature_table(panel, segment_categories=None):
    """
    Full feature pipeline on a speed panel.

    Returns (featured_df, segment_categories).
    """
    df = add_cyclical_time_features(panel)
    df, segment_categories = encode_segments(df, segment_categories)
    return df, segment_categories


def split_baseline_data(featured_df):
    """
    Prepare the baseline training set.

    CRITICAL: train ONLY on non-event days, so the baseline learns *normal*
    traffic and never absorbs event effects.

    Uses a TIME-BASED split (not random) to avoid leaking the future.

    Returns dict with X_train, X_test, y_train, y_test, and the cutoff date.
    """
    non_event = featured_df[~featured_df["is_event_day"]].copy()
    non_event["date_dt"] = pd.to_datetime(non_event["date"])

    cutoff = non_event["date_dt"].quantile(TRAIN_TEST_SPLIT_QUANTILE)
    train_mask = non_event["date_dt"] <= cutoff

    X = non_event[BASELINE_FEATURES]
    y = non_event[BASELINE_TARGET]

    return {
        "X_train": X[train_mask],
        "X_test":  X[~train_mask],
        "y_train": y[train_mask],
        "y_test":  y[~train_mask],
        "cutoff":  cutoff,
        "n_excluded_event_rows": int(featured_df["is_event_day"].sum()),
    }


if __name__ == "__main__":
    from road_network import build_road_graph
    from traffic_data import build_speed_panel

    G, seg_info = build_road_graph()
    panel = build_speed_panel(G, seg_info)
    featured, cats = build_feature_table(panel)
    split = split_baseline_data(featured)

    print(f"Feature columns: {BASELINE_FEATURES}")
    print(f"Train rows: {len(split['X_train']):,}")
    print(f"Test rows:  {len(split['X_test']):,}")
    print(f"Excluded event-day rows: {split['n_excluded_event_rows']:,}")
    print(f"Split cutoff date: {split['cutoff'].date()}")
