"""
============================================================================
EVENTS CALENDAR — derive REAL event days from the ASTraM CSV
============================================================================
Replaces the old hard-coded EVENT_DAYS dict. The WHEN / WHERE / TYPE of every
event now comes from the real data. Only crowd size is augmented (the CSV has
crowd numbers for just 2 of 327 events), using a documented, defensible prior
table keyed by event type and venue.

This is the honest version: real event calendar + stated crowd assumptions,
never invented dates or locations.
"""
import re
import numpy as np
import pandas as pd

from data_loader import load_clean_events, GATHERING_CAUSES


# ---------------------------------------------------------------------------
# CROWD PRIORS  (documented assumptions — cite these in the presentation)
# Based on public venue capacities and typical Bengaluru event sizes.
# Used ONLY where the CSV has no explicit crowd number.
# ---------------------------------------------------------------------------
CROWD_PRIORS = {
    "public_event": 30000,   # IPL/cricket dominates; Chinnaswamy cap ~40k
    "procession":   3000,    # religious processions / pallakki utsava
    "vip_movement": 1000,    # small crowd, high disruption (closures)
    "protest":      2000,
    "congestion":   5000,    # generic congestion reports
    "construction": 0,       # not crowd-driven; impact via lane closure
}

# Venue-specific overrides (when description names a known high-capacity venue)
VENUE_CROWD_OVERRIDE = [
    (r"chinnaswamy|ipl|rcb|cricket", 38000),   # stadium events
    (r"marathon|tcs",                15000),
]


def extract_crowd_from_text(text):
    """Pull an explicit crowd number from description text, if present."""
    if pd.isna(text):
        return None
    m = re.search(r"(\d{2,6})\s*(?:persons?|people|attendees|pax|crowd)",
                  str(text), re.I)
    return int(m.group(1)) if m else None


def estimate_crowd(row):
    """
    Crowd size for one event:
      1. explicit number in text (best)
      2. venue-specific override by keyword
      3. event-type prior (fallback)
    Returns (crowd, source) so we can report how each was derived.
    """
    explicit = extract_crowd_from_text(row.get("description"))
    if explicit:
        return explicit, "text"

    desc = str(row.get("description", "")).lower()
    for pattern, crowd in VENUE_CROWD_OVERRIDE:
        if re.search(pattern, desc):
            return crowd, "venue_override"

    return CROWD_PRIORS.get(row["event_cause"], 3000), "type_prior"


def derive_event_days(clean_events=None, zone_only=True,
                      causes=None, min_priority=None):
    """
    Build the event calendar from the real ASTraM data.

    Returns a dict {date_str: {type, cause, junction, crowd, crowd_source,
                               start_h, end_h, n_events, lat, lng}}
    keyed by date. If multiple events share a date, the highest-impact one
    (road-closure first, then priority) represents that day.
    """
    if clean_events is None:
        clean_events = load_clean_events(zone_only=zone_only)

    causes = causes or (GATHERING_CAUSES + ["construction"])
    relevant = clean_events[clean_events["event_cause"].isin(causes)].copy()
    relevant = relevant[relevant["start_ist"].notna()]

    # Rank within a day: road closures and high priority first
    relevant["closure_rank"] = relevant["requires_road_closure"].fillna(False).astype(int)
    relevant["prio_rank"] = (relevant["priority"] == "High").astype(int)

    event_days = {}
    for date_str, group in relevant.groupby("date"):
        # pick the representative (most disruptive) event of the day
        rep = group.sort_values(["closure_rank", "prio_rank"], ascending=False).iloc[0]

        crowd, crowd_src = estimate_crowd(rep)
        start_h = int(rep["start_ist"].hour)
        dur = rep.get("duration_h")
        end_h = int(min(23, start_h + (dur if pd.notna(dur) and dur > 0 else 3)))

        event_days[date_str] = {
            "type":         rep["event_cause"],
            "cause":        rep["event_cause"],
            "junction":     rep["nearest_junction"],
            "crowd":        int(crowd),
            "crowd_source": crowd_src,
            "start_h":      start_h,
            "end_h":        end_h,
            "n_events":     len(group),
            "lat":          float(rep["latitude"]),
            "lng":          float(rep["longitude"]),
        }
    return event_days


def summarize_calendar(event_days):
    """Print a quick EDA summary of the derived calendar."""
    df = pd.DataFrame(event_days).T
    print(f"Derived {len(df)} event-days from real ASTraM data")
    print(f"\nBy type:")
    print(df["type"].value_counts().to_string())
    print(f"\nCrowd source (how crowd size was determined):")
    print(df["crowd_source"].value_counts().to_string())
    return df


if __name__ == "__main__":
    # Zone-only by default; pass zone_only=False for the whole city
    events = load_clean_events(zone_only=False)
    cal = derive_event_days(events, zone_only=False)
    summarize_calendar(cal)

    print(f"\nSample event-days:")
    for date, ev in list(cal.items())[:8]:
        print(f"  {date}: {ev['type']:<14} @ {ev['junction']:<18} "
              f"crowd~{ev['crowd']:>6} ({ev['crowd_source']}) "
              f"{ev['start_h']:02d}:00-{ev['end_h']:02d}:00")
