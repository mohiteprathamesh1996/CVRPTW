"""
stochastic_analysis.py
-----------------------
Solves the CVRPTW under three travel-time scenarios (optimistic, most-likely,
pessimistic) and compares resulting KPIs.

The stochastic framing treats travel time as a three-point (PERT) random
variable parameterised by

    a = optimistic,  m = most-likely,  b = pessimistic

The PERT mean and variance are:
    μ  = (a + 4m + b) / 6
    σ² = ((b - a) / 6)²

These are computed arc-by-arc for downstream robustness analysis.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_loader import CVRPTWData, SCENARIOS
from src.cvrptw_solver import (
    CVRPTWSolver,
    SolverResult,
    DEFAULT_NUM_VEHICLES,
    DEFAULT_MAX_WEIGHT_KG,
    DEFAULT_MAX_VOLUME_M3,
)


# ─────────────────────────────────────────────────────────────────────────────
# PERT statistics
# ─────────────────────────────────────────────────────────────────────────────
def compute_pert_statistics(data: CVRPTWData) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute arc-level PERT mean and variance from the three time matrices.

    Returns
    -------
    mu_matrix  : (N×N) array of PERT mean travel times
    var_matrix : (N×N) array of PERT variances
    """
    a = data.time_matrices["optimistic"].astype(float)
    m = data.time_matrices["mostlikely"].astype(float)
    b = data.time_matrices["pessimistic"].astype(float)

    mu_matrix  = (a + 4 * m + b) / 6.0
    var_matrix = ((b - a) / 6.0) ** 2

    return mu_matrix, var_matrix


def compute_arc_uncertainty_ratio(data: CVRPTWData) -> np.ndarray:
    """
    Uncertainty ratio = (pessimistic − optimistic) / most-likely
    for each arc. High values indicate arcs with volatile travel times.
    """
    a   = data.time_matrices["optimistic"].astype(float)
    m   = data.time_matrices["mostlikely"].astype(float)
    b   = data.time_matrices["pessimistic"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(m > 0, (b - a) / m, 0.0)
    return ratio


# ─────────────────────────────────────────────────────────────────────────────
# Multi-scenario solve
# ─────────────────────────────────────────────────────────────────────────────
def solve_all_scenarios(
    data: CVRPTWData,
    num_vehicles: int = DEFAULT_NUM_VEHICLES,
    max_weight_kg: float = DEFAULT_MAX_WEIGHT_KG,
    max_volume_m3: float = DEFAULT_MAX_VOLUME_M3,
    time_limit_s: int = 60,
) -> Dict[str, SolverResult]:
    """
    Solve CVRPTW under all three travel-time scenarios and return results.

    Each solve uses the SAME fleet and demand data but a different travel-time
    matrix, mimicking how the routing decision responds to traffic conditions.
    """
    results: Dict[str, SolverResult] = {}
    for scenario in SCENARIOS:
        solver = CVRPTWSolver(
            data=data,
            scenario=scenario,
            num_vehicles=num_vehicles,
            max_weight_kg=max_weight_kg,
            max_volume_m3=max_volume_m3,
            time_limit_s=time_limit_s,
        )
        result = solver.solve()
        results[scenario] = result
        print(f"  [{scenario:12s}]  status={result.status}  "
              f"time={result.total_travel_time:.1f}min  "
              f"late={result.late_deliveries}  "
              f"vehicles={result.num_vehicles_used}  "
              f"({result.wall_time_s:.1f}s)")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Comparison tables
# ─────────────────────────────────────────────────────────────────────────────
def scenario_kpi_table(results: Dict[str, SolverResult]) -> pd.DataFrame:
    """Build a KPI comparison DataFrame across scenarios."""
    rows = []
    for scen in SCENARIOS:
        r = results.get(scen)
        if r is None:
            continue
        rows.append({
            "Scenario":           scen.capitalize(),
            "Status":             r.status,
            "Vehicles Used":      r.num_vehicles_used,
            "Total Travel (min)": round(r.total_travel_time, 1),
            "Total Distance (km)":round(r.total_distance_km, 1),
            "Late Deliveries":    r.late_deliveries,
            "Solve Time (s)":     round(r.wall_time_s, 2),
        })
    return pd.DataFrame(rows).set_index("Scenario")


def route_overlap_matrix(results: Dict[str, SolverResult]) -> pd.DataFrame:
    """
    Compute pairwise route-sequence overlap between scenarios.
    Overlap = fraction of customer-pairs that appear consecutively in both.
    """
    def consecutive_pairs(routes: List[List[int]]) -> set:
        pairs = set()
        for route in routes:
            customers = [n for n in route if n != 0]
            for a, b in zip(customers, customers[1:]):
                pairs.add((a, b))
        return pairs

    scenario_pairs = {
        s: consecutive_pairs(r.routes)
        for s, r in results.items()
        if r.is_solved()
    }
    solved = list(scenario_pairs.keys())
    mat = pd.DataFrame(index=solved, columns=solved, dtype=float)
    for s1 in solved:
        for s2 in solved:
            p1, p2 = scenario_pairs[s1], scenario_pairs[s2]
            union = p1 | p2
            inter = p1 & p2
            mat.loc[s1, s2] = len(inter) / len(union) if union else 1.0
    return mat


def lateness_by_customer(
    data: CVRPTWData,
    results: Dict[str, SolverResult],
) -> pd.DataFrame:
    """
    For each customer, report estimated lateness (actual arrival − LAT) per
    scenario.  Positive = late, 0 = on-time or early.
    """
    rows: Dict[int, Dict] = {}
    for scen in SCENARIOS:
        r = results.get(scen)
        if r is None or not r.is_solved():
            continue
        for route, arrivals in zip(r.routes, r.arrival_times):
            for node, arr in zip(route, arrivals):
                if node == 0:
                    continue
                lat = int(data.lat[node])
                lateness = max(0.0, arr - lat)
                if node not in rows:
                    rows[node] = {"NODE_ID": node}
                rows[node][scen] = round(lateness, 1)

    df = pd.DataFrame(rows.values()).set_index("NODE_ID").sort_index()
    for scen in SCENARIOS:
        if scen not in df.columns:
            df[scen] = 0.0
    df["max_lateness"] = df[list(SCENARIOS)].max(axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate summary across multiple days
# ─────────────────────────────────────────────────────────────────────────────
def multi_day_scenario_summary(
    day_results: Dict[int, Dict[str, SolverResult]],
) -> pd.DataFrame:
    """
    Aggregate KPIs across all days and scenarios into a flat DataFrame.
    """
    records = []
    for day, scenario_results in day_results.items():
        for scen, r in scenario_results.items():
            records.append({
                "day":               day,
                "scenario":          scen,
                "status":            r.status,
                "vehicles_used":     r.num_vehicles_used,
                "total_travel_min":  r.total_travel_time,
                "total_dist_km":     r.total_distance_km,
                "late_deliveries":   r.late_deliveries,
            })
    return pd.DataFrame(records)


if __name__ == "__main__":
    from src.data_loader import load_day

    print("Loading Day 1 data …")
    data = load_day(1)
    print("Solving all scenarios …")
    results = solve_all_scenarios(data, time_limit_s=30)
    print()
    print(scenario_kpi_table(results).to_string())
    print()
    print("Route overlap:")
    print(route_overlap_matrix(results).to_string())
