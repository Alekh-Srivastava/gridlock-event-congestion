"""
============================================================================
IMPACT — graph-hop weighted congestion propagation
============================================================================
When an event happens at a junction, its congestion does NOT spread by
straight-line distance — it spreads through the road network, junction by
junction. This module computes, for an event at any junction, how strongly
each road segment is affected.

Weight =  hop_decay(distance_in_hops)  x  capacity_adjustment
  - hop 0 (the event's own roads): full weight
  - each hop outward: weaker, per HOP_DECAY
  - narrow roads (low capacity): boosted, because they choke first
"""
import networkx as nx

from config import HOP_DECAY, MAX_HOPS, CAPACITY_ADJUSTMENT


def hop_distance_map(G, source, max_hops=MAX_HOPS):
    """
    BFS outward from `source` junction (treating the graph as undirected for
    propagation, since congestion backs up both ways).

    Returns {junction: hop_distance}.
    """
    visited = {source: 0}
    queue = [source]
    while queue:
        node = queue.pop(0)
        if visited[node] >= max_hops:
            continue
        neighbors = set(G.successors(node)) | set(G.predecessors(node))
        for nb in neighbors:
            if nb not in visited:
                visited[nb] = visited[node] + 1
                queue.append(nb)
    return visited


def segment_impact_weights(G, seg_info, event_junction):
    """
    Compute the impact weight for every segment given an event at a junction.

    Returns
    -------
    dict : {segment_id: {"weight": float, "hop": int, "road": str}}
           Only segments within MAX_HOPS are included.
    """
    hop_map = hop_distance_map(G, event_junction)
    max_cap = max(s["capacity"] for s in seg_info.values())

    weights = {}
    for sid, info in seg_info.items():
        min_hop = min(hop_map.get(info["u"], 999), hop_map.get(info["v"], 999))
        if min_hop > MAX_HOPS:
            continue
        base = HOP_DECAY.get(min_hop, 0.0)
        cap_ratio = info["capacity"] / max_cap
        weight = base * (1 - CAPACITY_ADJUSTMENT * cap_ratio)
        weights[sid] = {"weight": round(weight, 3), "hop": min_hop, "road": info["road"]}
    return weights


def segment_centrality(G, seg_info, k_approx=200):
    """
    Betweenness centrality per segment = average of its two endpoints'
    node betweenness. High centrality => critical bottleneck => priority.

    Uses approximate betweenness (k random sources) on large graphs to avoid
    the O(VE) hang that occurs with the real OSMnx graph (2776 nodes × 6643
    edges = ~39 s exact vs ~1 s approximate at k=200).
    """
    n = G.number_of_nodes()
    k = min(k_approx, n) if n > k_approx else None  # None = exact (small graphs)
    node_bc = nx.betweenness_centrality(G, k=k, seed=42)
    seg_bc = {}
    for sid, info in seg_info.items():
        seg_bc[sid] = (node_bc.get(info["u"], 0) + node_bc.get(info["v"], 0)) / 2
    return seg_bc


if __name__ == "__main__":
    from road_network import build_road_graph
    G, seg_info = build_road_graph()

    print("Impact weights for event at Chinnaswamy:\n")
    weights = segment_impact_weights(G, seg_info, "Chinnaswamy")
    for sid, w in sorted(weights.items(), key=lambda x: -x[1]["weight"])[:12]:
        print(f"  {w['road']:<18} hop={w['hop']} weight={w['weight']:.3f}")

    print("\nTop-centrality segments (bottlenecks):\n")
    bc = segment_centrality(G, seg_info)
    for sid, c in sorted(bc.items(), key=lambda x: -x[1])[:5]:
        print(f"  {seg_info[sid]['road']:<18} centrality={c:.3f}")
