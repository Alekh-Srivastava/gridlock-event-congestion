"""
============================================================================
DATA LOADER — import and clean the raw ASTraM event data
============================================================================
This is step 1 of the pipeline: take the messy 46-column anonymized CSV and
turn it into a clean, typed, filtered events table we can actually use.

Responsibilities:
  1. Load the raw CSV
  2. Parse timestamps (the data is UTC; Bengaluru is UTC+5:30)
  3. Drop rows with broken coordinates
  4. Filter to events inside our study zone
  5. Snap each event to its nearest junction
  6. Compute event duration where possible
  7. Return a tidy events DataFrame
"""
import numpy as np
import pandas as pd

from config import ASTRAM_CSV, VENUE_LAT, VENUE_LNG, ZONE_RADIUS_M
from road_network import JUNCTIONS, haversine_m, build_junction_coords


# Columns we actually keep from the 46 raw ones
USEFUL_COLUMNS = [
    "id", "event_type", "event_cause", "priority", "requires_road_closure",
    "latitude", "longitude", "corridor", "junction", "zone", "police_station",
    "start_datetime", "end_datetime", "created_date", "status",
    "veh_type", "description",
]

# Event causes that represent crowd-driven "gatherings" (the problem focus)
GATHERING_CAUSES = ["public_event", "procession", "vip_movement", "protest", "congestion"]


def load_raw_events(csv_path=ASTRAM_CSV):
    """Load the raw ASTraM CSV and keep only useful columns."""
    df = pd.read_csv(csv_path)
    # Keep only columns that exist (defensive — schema may vary)
    cols = [c for c in USEFUL_COLUMNS if c in df.columns]
    df = df[cols].copy()
    print(f"  Loaded {len(df):,} raw events, {len(cols)} columns kept")
    return df


def parse_timestamps(df):
    """Parse UTC timestamps and add IST-localised helper columns."""
    for col in ["start_datetime", "end_datetime", "created_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # IST versions for human-readable temporal features
    df["start_ist"]   = df["start_datetime"].dt.tz_convert("Asia/Kolkata")
    df["hour"]        = df["start_ist"].dt.hour + df["start_ist"].dt.minute / 60
    df["weekday"]     = df["start_ist"].dt.day_name()
    df["date"]        = df["start_ist"].dt.date.astype("string")

    # Duration (only available when end_datetime is present)
    df["duration_h"] = (df["end_datetime"] - df["start_datetime"]).dt.total_seconds() / 3600
    return df


def clean_coordinates(df):
    """Drop rows with missing or out-of-range coordinates (Bengaluru bbox)."""
    before = len(df)
    df = df[df["latitude"].between(12.7, 13.3) & df["longitude"].between(77.3, 77.8)].copy()
    print(f"  Dropped {before - len(df):,} rows with bad coordinates")
    return df


def filter_to_zone(df, venue_lat=VENUE_LAT, venue_lng=VENUE_LNG, radius_m=ZONE_RADIUS_M):
    """Keep only events within radius_m of the venue."""
    df["dist_to_venue_m"] = df.apply(
        lambda r: haversine_m(venue_lat, venue_lng, r["latitude"], r["longitude"]),
        axis=1,
    )
    in_zone = df[df["dist_to_venue_m"] <= radius_m].copy()
    print(f"  {len(in_zone):,} events within {radius_m}m of venue")
    return in_zone


def snap_to_nearest_junction(df, junctions=None):
    """Attach the nearest road-graph junction to each event.

    Vectorised with NumPy broadcasting — ~100× faster than the previous
    row-by-row Python loop, which blocked for minutes on large OSMnx graphs.
    """
    if junctions is None:
        junctions = JUNCTIONS
    names  = np.array(list(junctions.keys()))
    coords = np.array([junctions[n] for n in names])   # (N, 2) lat/lng

    lats = df["latitude"].values[:, None]   # (M, 1)
    lngs = df["longitude"].values[:, None]  # (M, 1)

    R    = 6_371_000.0
    lat1 = np.radians(lats)
    lng1 = np.radians(lngs)
    lat2 = np.radians(coords[:, 0])         # (N,)
    lng2 = np.radians(coords[:, 1])         # (N,)

    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a    = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    dists = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))   # (M, N)

    idx = np.argmin(dists, axis=1)
    df["nearest_junction"] = names[idx]
    df["snap_dist_m"]      = np.round(dists[np.arange(len(df)), idx], 1)
    return df


def load_clean_events(csv_path=ASTRAM_CSV, zone_only=True, G=None):
    """
    Full cleaning pipeline. Returns a tidy events DataFrame.

    Parameters
    ----------
    zone_only : bool
        If True, keep only events within ZONE_RADIUS_M of the venue.
    G : nx.Graph, optional
        Road graph. If provided (OSMnx mode), junction snapping uses real
        OSM node IDs. If None, falls back to the hardcoded JUNCTIONS dict.
    """
    print("[data_loader] Importing and cleaning ASTraM events...")
    df = load_raw_events(csv_path)
    df = parse_timestamps(df)
    df = clean_coordinates(df)
    if zone_only:
        df = filter_to_zone(df)
    junction_lookup = build_junction_coords(G) if G is not None else None
    df = snap_to_nearest_junction(df, junction_lookup)
    df["is_gathering"] = df["event_cause"].isin(GATHERING_CAUSES)
    print(f"[data_loader] Done. {len(df):,} clean events "
          f"({df['is_gathering'].sum()} gatherings).")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    events = load_clean_events()
    print("\nEvent cause breakdown (in zone):")
    print(events["event_cause"].value_counts().head(10).to_string())
    print("\nSample gathering events:")
    g = events[events["is_gathering"]].head(5)
    for _, r in g.iterrows():
        print(f"  [{r['event_cause']}] near {r['nearest_junction']} "
              f"({r['snap_dist_m']:.0f}m): {str(r['description'])[:50]}")
