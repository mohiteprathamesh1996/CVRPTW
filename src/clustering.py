"""
clustering.py
-------------
Clustering-based decomposition strategy for CVRPTW.

Approach
--------
1. Represent each customer node by its row in the travel-time matrix (most-
   likely) — a position-agnostic proximity embedding in time space.
2. Apply k-means to partition customers into k geographic/temporal clusters.
3. Solve an independent CVRPTW per cluster (smaller sub-problems → faster).
4. Compare cluster-decomposed solution against the global monolithic solve.

This is a classic matheuristic technique: decompose → solve subproblems →
reassemble.  It trades global optimality for tractability on large instances.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.data_loader import CVRPTWData, DEPOT_NODE
from src.cvrptw_solver import (
    CVRPTWSolver,
    SolverResult,
    DEFAULT_MAX_WEIGHT_KG,
    DEFAULT_MAX_VOLUME_M3,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_node_features(data: CVRPTWData, scenario: str = "mostlikely") -> np.ndarray:
    """
    Build a feature matrix for clustering.

    Features per node (excluding depot):
      - travel time FROM depot          (proximity measure)
      - travel time TO depot
      - mean travel time to all others  (centrality)
      - EAT, LAT                        (temporal context)

    Returns
    -------
    X : (n_customers, n_features) array
    """
    tm    = data.time_matrices[scenario].astype(float)  # (N+1)×(N+1)
    n     = data.num_customers
    cust  = np.arange(1, n + 1)                         # node indices 1..n

    # Distance from / to depot
    from_depot = tm[DEPOT_NODE, cust]    # row 0, cols 1..n
    to_depot   = tm[cust, DEPOT_NODE]    # col 0, rows 1..n

    # Mean travel to all other customers
    sub = tm[np.ix_(cust, cust)]         # n×n submatrix
    np.fill_diagonal(sub, np.nan)
    mean_to_others = np.nanmean(sub, axis=1)

    eat = data.eat[cust].astype(float)
    lat = data.lat[cust].astype(float)

    X = np.column_stack([from_depot, to_depot, mean_to_others, eat, lat])
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Optimal k selection
# ─────────────────────────────────────────────────────────────────────────────
def choose_k(
    X: np.ndarray,
    k_range: Optional[range] = None,
    random_state: int = 42,
) -> Tuple[int, pd.DataFrame]:
    """
    Evaluate silhouette scores for a range of k values and recommend the best.

    Returns
    -------
    best_k  : recommended number of clusters
    scores  : DataFrame with k and silhouette_score
    """
    if k_range is None:
        k_range = range(2, 9)

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    rows = []
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(Xs)
        sil    = silhouette_score(Xs, labels)
        rows.append({"k": k, "silhouette": round(sil, 4)})

    df     = pd.DataFrame(rows)
    best_k = int(df.loc[df["silhouette"].idxmax(), "k"])
    return best_k, df


# ─────────────────────────────────────────────────────────────────────────────
# Cluster assignment
# ─────────────────────────────────────────────────────────────────────────────
def cluster_customers(
    data: CVRPTWData,
    k: int,
    scenario: str = "mostlikely",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Assign each customer to one of k clusters.

    Returns
    -------
    DataFrame with columns: node_id, cluster, weight, volume, lat
    """
    X = extract_node_features(data, scenario)

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    km     = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(Xs)

    df = pd.DataFrame({
        "node_id": np.arange(1, data.num_customers + 1),
        "cluster": labels,
        "weight":  data.weights[1:],
        "volume":  data.volumes[1:],
        "lat":     data.lat[1:],
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Build a sub-problem CVRPTWData for a cluster
# ─────────────────────────────────────────────────────────────────────────────
def _build_cluster_data(
    data: CVRPTWData,
    cluster_nodes: List[int],
) -> CVRPTWData:
    """
    Create a reduced CVRPTWData containing only the depot (node 0) and the
    specified cluster_nodes.  Node IDs and matrix indices are remapped to a
    compact [0, 1, …, m] space.

    original index 0       → new index 0  (depot)
    cluster_nodes[0]       → new index 1
    cluster_nodes[1]       → new index 2  …
    """
    import copy

    all_nodes   = [DEPOT_NODE] + cluster_nodes
    n_sub       = len(all_nodes)
    orig_to_new = {orig: new for new, orig in enumerate(all_nodes)}

    # Subset orders
    sub_orders = data.orders[
        data.orders["NODE_ID"].isin(cluster_nodes)
    ].copy().reset_index(drop=True)

    # Subset matrices
    idx = np.array(all_nodes, dtype=int)
    sub_dist  = data.distance_matrix[np.ix_(idx, idx)]
    sub_times = {
        scen: data.time_matrices[scen][np.ix_(idx, idx)]
        for scen in data.time_matrices
    }

    sub_data = CVRPTWData(
        day            = data.day,
        orders         = sub_orders,
        distance_matrix= sub_dist,
        time_matrices  = sub_times,
        node_ids       = np.arange(n_sub, dtype=int),
    )
    return sub_data


# ─────────────────────────────────────────────────────────────────────────────
# Decomposed solve
# ─────────────────────────────────────────────────────────────────────────────
def solve_decomposed(
    data: CVRPTWData,
    k: int,
    scenario: str = "mostlikely",
    vehicles_per_cluster: int = 3,
    max_weight_kg: float = DEFAULT_MAX_WEIGHT_KG,
    max_volume_m3: float = DEFAULT_MAX_VOLUME_M3,
    time_limit_s: int = 30,
    random_state: int = 42,
) -> Tuple[Dict[int, SolverResult], pd.DataFrame]:
    """
    Cluster → solve each sub-problem → return per-cluster results.

    Parameters
    ----------
    k                   : number of clusters
    vehicles_per_cluster: vehicles allocated to each cluster
    """
    assignment  = cluster_customers(data, k, scenario, random_state)
    cluster_ids = sorted(assignment["cluster"].unique())

    results: Dict[int, SolverResult] = {}
    rows: List[Dict] = []

    for c in cluster_ids:
        nodes_in_cluster = assignment.loc[
            assignment["cluster"] == c, "node_id"
        ].tolist()
        print(f"  Cluster {c}: {len(nodes_in_cluster)} customers", end=" … ", flush=True)

        sub_data = _build_cluster_data(data, nodes_in_cluster)
        solver   = CVRPTWSolver(
            data          = sub_data,
            scenario      = scenario,
            num_vehicles  = vehicles_per_cluster,
            max_weight_kg = max_weight_kg,
            max_volume_m3 = max_volume_m3,
            time_limit_s  = time_limit_s,
        )
        r = solver.solve()
        results[c] = r
        print(f"status={r.status}  time={r.total_travel_time:.1f}min  late={r.late_deliveries}")
        rows.append({
            "cluster":          c,
            "n_customers":      len(nodes_in_cluster),
            "status":           r.status,
            "vehicles_used":    r.num_vehicles_used,
            "total_travel_min": round(r.total_travel_time, 1),
            "late_deliveries":  r.late_deliveries,
        })

    summary = pd.DataFrame(rows)
    return results, summary


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated comparison
# ─────────────────────────────────────────────────────────────────────────────
def compare_decomposed_vs_global(
    cluster_results: Dict[int, SolverResult],
    global_result: SolverResult,
) -> pd.DataFrame:
    """Compare aggregate decomposed stats against the global solve."""
    decomposed_travel = sum(r.total_travel_time for r in cluster_results.values() if r.is_solved())
    decomposed_late   = sum(r.late_deliveries   for r in cluster_results.values() if r.is_solved())
    decomposed_veh    = sum(r.num_vehicles_used for r in cluster_results.values() if r.is_solved())
    decomposed_dist   = sum(r.total_distance_km for r in cluster_results.values() if r.is_solved())

    df = pd.DataFrame([
        {
            "approach":         "Decomposed (k-means)",
            "total_travel_min": round(decomposed_travel, 1),
            "total_dist_km":    round(decomposed_dist, 1),
            "vehicles_used":    decomposed_veh,
            "late_deliveries":  decomposed_late,
        },
        {
            "approach":         "Global (monolithic)",
            "total_travel_min": round(global_result.total_travel_time, 1),
            "total_dist_km":    round(global_result.total_distance_km, 1),
            "vehicles_used":    global_result.num_vehicles_used,
            "late_deliveries":  global_result.late_deliveries,
        },
    ]).set_index("approach")

    gap = (decomposed_travel - global_result.total_travel_time) / max(1, global_result.total_travel_time) * 100
    df["optimality_gap_%"] = [round(gap, 2), 0.0]
    return df


if __name__ == "__main__":
    from src.data_loader import load_day
    from src.cvrptw_solver import solve as global_solve

    data          = load_day(1)
    global_result = global_solve(data, scenario="mostlikely", time_limit_s=30)

    X       = extract_node_features(data)
    best_k, sil_df = choose_k(X)
    print(f"Best k = {best_k}")
    print(sil_df.to_string(index=False))

    cluster_results, summary = solve_decomposed(data, k=best_k, time_limit_s=20)
    print("\nCluster summary:")
    print(summary.to_string(index=False))

    print("\nDecomposed vs Global:")
    print(compare_decomposed_vs_global(cluster_results, global_result).to_string())
