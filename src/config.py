"""
============================================================================
CONFIG — Central configuration for the whole project
============================================================================
Every tunable number lives here so you never hunt through code to change
a parameter. Import this module anywhere with:  from config import CONFIG
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# PATHS  (auto-resolve relative to this file, so the project runs anywhere)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR       = ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR         = ROOT / "models"
OUTPUTS_DIR        = ROOT / "outputs"

ASTRAM_CSV = RAW_DATA_DIR / "astram_events.csv"

# ---------------------------------------------------------------------------
# STUDY ZONE  (the venue + radius we model deeply)
# ---------------------------------------------------------------------------
VENUE_NAME   = "Chinnaswamy Stadium"
VENUE_LAT    = 12.9788
VENUE_LNG    = 77.5996
ZONE_RADIUS_M = 2000          # only events within this radius are "in zone"

# ---------------------------------------------------------------------------
# TIME BINNING
# ---------------------------------------------------------------------------
BIN_MINUTES   = 15            # size of each time window
BINS_PER_DAY  = 24 * 60 // BIN_MINUTES   # = 96

# ---------------------------------------------------------------------------
# BASELINE MODEL  (predicts NORMAL, non-event speed)
# ---------------------------------------------------------------------------
# Features the baseline is allowed to use. NOTE: no event info here —
# the baseline must never see events, so the delta stays clean.
BASELINE_FEATURES = [
    "segment_code",   # WHICH road (proxy for demand/context — see scalability note)
    "hour_sin", "hour_cos",        # time of day, cyclically encoded
    "weekday_sin", "weekday_cos",  # day of week, cyclically encoded
    "is_weekend",
    "is_rain",
    "lanes",
    "free_flow_speed",
    "capacity",
]
BASELINE_TARGET = "observed_speed"

LGBM_PARAMS = {
    "n_estimators":      300,
    "max_depth":         6,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "random_state":      42,
    "verbose":           -1,
}
TRAIN_TEST_SPLIT_QUANTILE = 0.8   # time-based: first 80% of dates train, rest test

# ---------------------------------------------------------------------------
# DELTA / IMPACT
# ---------------------------------------------------------------------------
NOISE_FLOOR_KMH = 3.0     # |delta| below this is treated as noise, not impact
SEVERE_DELTA    = -10.0   # km/h, classified "severely affected"
MODERATE_DELTA  = -5.0    # km/h, classified "moderately affected"

# ---------------------------------------------------------------------------
# GRAPH-HOP WEIGHTED IMPACT
# ---------------------------------------------------------------------------
MAX_HOPS = 6
HOP_DECAY = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.15, 4: 0.05, 5: 0.02, 6: 0.01}
CAPACITY_ADJUSTMENT = 0.4   # narrow roads absorb less → up to +40% impact weight

# ---------------------------------------------------------------------------
# SYNTHETIC DATA GENERATION  (stand-in for TomTom/Google until API is wired)
# ---------------------------------------------------------------------------
SYNTH_N_DAYS        = 150
SYNTH_START         = "2023-11-01"
SYNTH_RAIN_FRAC     = 0.20
SYNTH_SEED          = 42
# Cap for synthetic mode: only generate data for the N nearest segments to the
# venue. Real OSMnx graphs have 6,000+ segments; generating all × 150 days ×
# 96 bins = ~95M rows causes OOM.
# 100 nearest segments = ~1.4M rows — safe on systems with 4 GB RAM or less.
# Increase to 200-300 if you have 8+ GB free. Has no effect with real TomTom data.
SYNTH_MAX_SEGMENTS  = 100

# Real IPL / event days observed in the ASTraM data, used as synthetic events.
# (junction must match a node in the road graph)
EVENT_DAYS = {
    "2024-03-09": {"type": "cricket",      "junction": "Chinnaswamy",     "crowd": 35000, "start_h": 17, "end_h": 22},
    "2024-03-16": {"type": "cricket",      "junction": "Chinnaswamy",     "crowd": 40000, "start_h": 19, "end_h": 23},
    "2024-03-23": {"type": "cricket",      "junction": "Chinnaswamy",     "crowd": 38000, "start_h": 15, "end_h": 20},
    "2024-03-30": {"type": "cricket",      "junction": "Chinnaswamy",     "crowd": 42000, "start_h": 19, "end_h": 23},
    "2024-01-26": {"type": "procession",   "junction": "Queens_Circle",   "crowd": 5000,  "start_h": 10, "end_h": 14},
    "2024-02-15": {"type": "public_event", "junction": "MG_Road_Trinity", "crowd": 8000,  "start_h": 16, "end_h": 21},
}
