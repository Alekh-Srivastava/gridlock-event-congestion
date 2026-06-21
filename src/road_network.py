"""
============================================================================
ROAD NETWORK — junctions, segments, and the road graph
============================================================================
Default mode: OSMnx pulls the real Bengaluru road network from OpenStreetMap
in one call (hundreds of real junctions/segments with full metadata).

Offline fallback: set USE_HARDCODED_GRAPH = True for dev without internet.
The hardcoded 15-junction graph is kept here for that purpose only.
"""
import networkx as nx
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

from config import VENUE_LAT, VENUE_LNG, ZONE_RADIUS_M

# Set True for offline dev (no internet); False = real OSMnx (default)
USE_HARDCODED_GRAPH = False


# ---------------------------------------------------------------------------
# Offline fallback — real Bengaluru coordinates, hand-coded road topology
# ---------------------------------------------------------------------------
JUNCTIONS = {
    "Chinnaswamy":      (12.9788, 77.5996),
    "Cubbon_Park_Gate": (12.9770, 77.5920),
    "MG_Road_Trinity":  (12.9716, 77.6048),
    "Queens_Circle":    (12.9740, 77.5960),
    "Anil_Kumble_Cir":  (12.9810, 77.5990),
    "KR_Circle":        (12.9760, 77.5750),
    "Minsk_Square":     (12.9815, 77.5945),
    "Raj_Bhavan_Rd":    (12.9850, 77.5920),
    "Kasturba_Rd_Jn":   (12.9730, 77.5890),
    "Richmond_Circle":  (12.9670, 77.5960),
    "Hudson_Circle":    (12.9830, 77.6030),
    "Ulsoor_Gate":      (12.9810, 77.6090),
    "Brigade_Rd_Jn":    (12.9710, 77.6010),
    "Lavelle_Rd_Jn":    (12.9680, 77.5930),
    "Residency_Rd_Jn":  (12.9750, 77.6020),
}

SEGMENTS = [
    ("Cubbon_Park_Gate", "Queens_Circle",    "Kasturba Road",    3, 40),
    ("Queens_Circle",    "Chinnaswamy",      "MG Road",          4, 35),
    ("Chinnaswamy",      "MG_Road_Trinity",  "MG Road",          4, 35),
    ("MG_Road_Trinity",  "Chinnaswamy",      "MG Road",          4, 35),
    ("Chinnaswamy",      "Queens_Circle",    "MG Road",          4, 35),
    ("Queens_Circle",    "Cubbon_Park_Gate", "Kasturba Road",    3, 40),
    ("Queens_Circle",    "KR_Circle",        "Queen's Road",     3, 40),
    ("KR_Circle",        "Queens_Circle",    "Queen's Road",     3, 40),
    ("Anil_Kumble_Cir",  "Chinnaswamy",      "Cubbon Road",      3, 35),
    ("Chinnaswamy",      "Anil_Kumble_Cir",  "Cubbon Road",      3, 35),
    ("Minsk_Square",     "Anil_Kumble_Cir",  "Nrupathunga Road", 3, 40),
    ("Anil_Kumble_Cir",  "Hudson_Circle",    "Nrupathunga Road", 3, 40),
    ("Hudson_Circle",    "Ulsoor_Gate",      "Ulsoor Road",      3, 35),
    ("Ulsoor_Gate",      "Hudson_Circle",    "Ulsoor Road",      3, 35),
    ("Raj_Bhavan_Rd",    "Minsk_Square",     "Raj Bhavan Road",  2, 30),
    ("Kasturba_Rd_Jn",   "Cubbon_Park_Gate", "Kasturba Road",    3, 40),
    ("Richmond_Circle",  "Queens_Circle",    "Richmond Road",    3, 35),
    ("Queens_Circle",    "Richmond_Circle",  "Richmond Road",    3, 35),
    ("Richmond_Circle",  "Lavelle_Rd_Jn",    "Lavelle Road",     2, 30),
    ("Lavelle_Rd_Jn",    "Richmond_Circle",  "Lavelle Road",     2, 30),
    ("Brigade_Rd_Jn",    "MG_Road_Trinity",  "Brigade Road",     3, 30),
    ("MG_Road_Trinity",  "Brigade_Rd_Jn",    "Brigade Road",     3, 30),
    ("Residency_Rd_Jn",  "MG_Road_Trinity",  "Residency Road",   3, 35),
    ("Richmond_Circle",  "Brigade_Rd_Jn",    "Church Street",    2, 30),
]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def haversine_m(lat1, lng1, lat2, lng2):
    """Great-circle distance in metres between two lat/lng points."""
    R = 6_371_000
    dlat, dlng = radians(lat2 - lat1), radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ---------------------------------------------------------------------------
# OSM value parsers (OSM attributes can be strings, lists, or None)
# ---------------------------------------------------------------------------
def _parse_lanes(x, default=2):
    """Parse OSM lanes value to int. Handles strings, lists, None."""
    if x is None:
        return default
    if isinstance(x, list):
        try:
            return max(int(str(v).split(";")[0].strip()) for v in x if v)
        except (ValueError, TypeError):
            return default
    try:
        return max(1, int(str(x).split(";")[0].strip()))
    except (ValueError, TypeError):
        return default


def _parse_speed(x, default=40.0):
    """Parse OSM maxspeed to float km/h. Handles '50 mph', lists, None."""
    if x is None:
        return float(default)
    if isinstance(x, list):
        x = x[0] if x else None
        if x is None:
            return float(default)
    s = str(x).strip().lower()
    if "mph" in s:
        try:
            return round(float(s.replace("mph", "").strip()) * 1.60934, 1)
        except ValueError:
            return float(default)
    try:
        return float(s.replace("km/h", "").replace("kph", "").strip())
    except ValueError:
        return float(default)


def _first(x):
    """Return first element if list, else x. Ensures road names are strings."""
    if isinstance(x, list):
        return x[0] if x else "unnamed"
    return x if x else "unnamed"


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------
def _build_osmnx_graph(venue_lat, venue_lng, radius_m):
    """Download and build the real road graph from OpenStreetMap via OSMnx.

    The GraphML file is cached next to road_network.py so subsequent runs
    skip the ~10 s download and rebuild.
    """
    import osmnx as ox
    import joblib

    cache_path = Path(__file__).resolve().parent / "_osmnx_cache.graphml"
    seg_cache  = Path(__file__).resolve().parent / "_seg_info_cache.joblib"

    if cache_path.exists() and seg_cache.exists():
        print("  [road_network] Loading cached OSM graph (delete _osmnx_cache.graphml to refresh)...")
        G_multi = ox.load_graphml(cache_path)
    else:
        print(f"  [road_network] Downloading OSM drive network "
              f"({radius_m}m radius around {venue_lat:.4f}, {venue_lng:.4f})...")
        G_multi = ox.graph_from_point(
            (venue_lat, venue_lng), dist=radius_m, network_type="drive"
        )
        ox.save_graphml(G_multi, cache_path)
        print(f"  [road_network] Cached OSM graph to {cache_path.name}")

    # Normalise coordinates: OSMnx stores lat in "y", lng in "x"
    for _, d in G_multi.nodes(data=True):
        d["lat"] = d.get("y", 0.0)
        d["lng"] = d.get("x", 0.0)

    # Convert MultiDiGraph -> DiGraph (keeps shortest parallel edge per pair)
    G = ox.convert.to_digraph(G_multi, weight="length")

    # Ensure lat/lng survived the conversion
    for _, d in G.nodes(data=True):
        if "lat" not in d:
            d["lat"] = d.get("y", 0.0)
            d["lng"] = d.get("x", 0.0)

    if seg_cache.exists() and cache_path.exists():
        seg_info = joblib.load(seg_cache)
    else:
        seg_info = {}
        for u, v, d in G.edges(data=True):
            lanes = _parse_lanes(d.get("lanes"))
            ffs   = _parse_speed(d.get("maxspeed"))
            seg_id = f"{u}__{v}"
            seg_info[seg_id] = {
                "u": u, "v": v,
                "road":     _first(d.get("name", "unnamed")),
                "lanes":    lanes,
                "ffs":      ffs,
                "capacity": lanes * ffs,
                "length_m": float(d.get("length", 0.0)),
            }
        joblib.dump(seg_info, seg_cache)

    return G, seg_info


def _build_hardcoded_graph():
    """Build the 15-junction offline fallback graph."""
    G = nx.DiGraph()
    for name, (lat, lng) in JUNCTIONS.items():
        G.add_node(name, lat=lat, lng=lng)

    for u, v, road, lanes, ffs in SEGMENTS:
        lat1, lng1 = JUNCTIONS[u]
        lat2, lng2 = JUNCTIONS[v]
        length = haversine_m(lat1, lng1, lat2, lng2)
        G.add_edge(u, v,
                   segment_id=f"{u}__{v}", road_name=road,
                   lanes=lanes, free_flow_speed=ffs,
                   capacity=lanes * ffs, length_m=round(length, 1))

    seg_info = {}
    for u, v, d in G.edges(data=True):
        seg_info[d["segment_id"]] = {
            "u": u, "v": v,
            "road":     d["road_name"],
            "lanes":    d["lanes"],
            "ffs":      d["free_flow_speed"],
            "capacity": d["capacity"],
            "length_m": d["length_m"],
        }
    return G, seg_info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_road_graph(venue_lat=VENUE_LAT, venue_lng=VENUE_LNG, radius_m=ZONE_RADIUS_M):
    """
    Build the directed road graph.

    Returns
    -------
    G : nx.DiGraph
        Nodes carry lat/lng attributes. Edges carry road metadata.
    seg_info : dict
        {segment_id: {u, v, road, lanes, ffs, capacity, length_m}}
    """
    if USE_HARDCODED_GRAPH:
        print("  [road_network] OFFLINE mode — using hardcoded 15-junction graph.")
        return _build_hardcoded_graph()

    try:
        return _build_osmnx_graph(venue_lat, venue_lng, radius_m)
    except Exception as e:
        print(f"  [road_network] OSMnx failed ({e}). Falling back to hardcoded graph.")
        return _build_hardcoded_graph()


def build_junction_coords(G):
    """
    Extract {node: (lat, lng)} from any graph (OSMnx or hardcoded).
    Used by data_loader to snap events to the nearest junction.
    """
    return {n: (d["lat"], d["lng"]) for n, d in G.nodes(data=True)
            if d.get("lat") is not None}


if __name__ == "__main__":
    G, seg_info = build_road_graph()
    print(f"\nRoad graph: {G.number_of_nodes()} junctions, {G.number_of_edges()} segments")

    print("\nSample segments:")
    for sid, info in list(seg_info.items())[:6]:
        print(f"  [{info['road']:<22}] lanes={info['lanes']} "
              f"ffs={info['ffs']} cap={info['capacity']} "
              f"len={info['length_m']:.0f}m")

    missing = [n for n, d in G.nodes(data=True) if "lat" not in d]
    print(f"\nNodes missing lat/lng: {len(missing)} (want 0)")
