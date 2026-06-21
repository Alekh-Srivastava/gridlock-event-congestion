"""
============================================================================
BASELINE MODEL — learns NORMAL (non-event) traffic speed
============================================================================
The baseline answers: "what would the speed on this segment be at this hour,
this weekday, this weather — IF NO EVENT were happening?"

It is trained ONLY on non-event data. On an event day we run it to get the
counterfactual (the world without the event), and the gap between that and
the real measured speed is the event's impact (the delta).

Train once, reuse for months. Retrain only when roads physically change.
"""
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import LGBM_PARAMS, BASELINE_FEATURES, MODELS_DIR


class BaselineModel:
    """Wraps a LightGBM regressor plus its segment encoding."""

    def __init__(self):
        self.model = None
        self.segment_categories = None
        self.metrics = {}

    def train(self, split, segment_categories):
        """Fit on the non-event training split."""
        self.segment_categories = segment_categories
        self.model = lgb.LGBMRegressor(**LGBM_PARAMS)
        self.model.fit(split["X_train"], split["y_train"])

        # Evaluate on the held-out (later in time) non-event test set
        train_pred = self.model.predict(split["X_train"])
        test_pred  = self.model.predict(split["X_test"])
        self.metrics = {
            "train_mae":  float(mean_absolute_error(split["y_train"], train_pred)),
            "test_mae":   float(mean_absolute_error(split["y_test"], test_pred)),
            "test_rmse":  float(np.sqrt(mean_squared_error(split["y_test"], test_pred))),
            "train_rows": int(len(split["X_train"])),
            "test_rows":  int(len(split["X_test"])),
        }
        return self.metrics

    def predict(self, featured_df):
        """Predict the normal (counterfactual) speed for any featured rows."""
        return self.model.predict(featured_df[BASELINE_FEATURES])

    def feature_importance(self):
        return dict(sorted(
            zip(BASELINE_FEATURES, self.model.feature_importances_),
            key=lambda x: -x[1],
        ))

    def save(self, path=None):
        path = path or (MODELS_DIR / "baseline_model.joblib")
        joblib.dump(
            {"model": self.model,
             "segment_categories": self.segment_categories,
             "metrics": self.metrics},
            path,
        )
        return path

    @classmethod
    def load(cls, path=None):
        path = path or (MODELS_DIR / "baseline_model.joblib")
        blob = joblib.load(path)
        obj = cls()
        obj.model = blob["model"]
        obj.segment_categories = blob["segment_categories"]
        obj.metrics = blob["metrics"]
        return obj


if __name__ == "__main__":
    from road_network import build_road_graph
    from traffic_data import build_speed_panel
    from features import build_feature_table, split_baseline_data

    G, seg_info = build_road_graph()
    panel = build_speed_panel(G, seg_info)
    featured, cats = build_feature_table(panel)
    split = split_baseline_data(featured)

    bm = BaselineModel()
    metrics = bm.train(split, cats)
    print(f"Baseline trained:")
    print(f"  Test MAE:  {metrics['test_mae']:.2f} km/h")
    print(f"  Test RMSE: {metrics['test_rmse']:.2f} km/h")
    print(f"\nFeature importance:")
    for f, imp in bm.feature_importance().items():
        print(f"  {f:<18} {imp}")
