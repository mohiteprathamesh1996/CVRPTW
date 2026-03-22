"""
robustness_analysis.py
-----------------------
Evaluates routing solution stability under travel-time uncertainty.

Two complementary analyses are performed:

1. Route-level robustness
   For the most-likely solution, apply optimistic and pessimistic travel times
   and measure induced arrival-time deviation without re-solving.

2. Customer-level delay risk
   Using PERT-derived arc variances, propagate variance along each route to
   estimate per-customer arrival-time standard deviation (σ_arrival).
   A customer is classified as "high risk" if the probability of exceeding
   its LAT under a normal approximation exceeds a threshold.

3. Arc sensitivity
   Identify arcs whose uncertainty ratio contributes most to route instability.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data_loader import CVRPTWData, DEPOT_NODE, SCENARIOS
from src.cvrptw_solver import SolverResult
from src.stochastic_analysis import compute_pert_statistics, compute_arc_uncertainty_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _simulate_route_arrivals(
    route: List[int],
    time_matrix: np.ndarray,
    service_times: np.ndarray,
    depot_departure: float = 0.0,
) -> List[float]:
    """
    Forward-pass arrival time simulation along a single route under a given
    time matrix.  Waiting at time windows is respected (arrive early → wait).
    Returns arrival times at each node (depot included at position 0).
    """
    arrivals = [depot_departure]
    current_time = depot_departure
    for k in range(1, len(route)):
        i, j = route[k - 1], route[k]
        travel = time_matrix[i][j]
        svc    = service_times[i]
        current_time += svc + travel
        arrivals.append(current_time)
    return arrivals


def _propagate_variance(
    route: List[int],
    var_matrix: np.ndarray,
    service_times: np.ndarray,
) -> List[float]:
    """
    Propagate arc-level PERT variance along a route.
    Variance accumulates additively (independence assumption).
    Returns cumulative variance at each node.
    """
    cum_vars = [0.0]
    cum_var  = 0.0
    for k in range(1, len(route)):
        i, j = route[k - 1], route[k]
        cum_var += var_matrix[i][j]
        cum_vars.append(cum_var)
    return cum_vars


# ─────────────────────────────────────────────────────────────────────────────
# 1. Route-level robustness (cross-scenario arrival deviation)
# ─────────────────────────────────────────────────────────────────────────────
def route_arrival_deviation(
    result: SolverResult,
    data: CVRPTWData,
) -> pd.DataFrame:
    """
    Fix routes from the most-likely solution; re-simulate arrivals under all
    three time matrices.  Returns per-customer arrival-time spread.
    """
    records = []
    for route_idx, route in enumerate(result.routes):
        for scen in SCENARIOS:
            arrivals = _simulate_route_arrivals(
                route,
                data.time_matrices[scen],
                data.service_times,
            )
            for pos, (node, arr) in enumerate(zip(route, arrivals)):
                if node == DEPOT_NODE:
                    continue
                records.append({
                    "route":    route_idx + 1,
                    "position": pos,
                    "node":     node,
                    "scenario": scen,
                    "arrival":  arr,
                    "lat":      int(data.lat[node]),
                    "late":     max(0.0, arr - data.lat[node]),
                })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    pivot = df.pivot_table(
        index=["route", "node"],
        columns="scenario",
        values="arrival",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    pivot["arrival_spread"] = (
        pivot.get("pessimistic", 0) - pivot.get("optimistic", 0)
    )
    pivot["lat"] = pivot["node"].apply(lambda n: int(data.lat[n]))
    pivot["late_in_pessimistic"] = (
        pivot.get("pessimistic", 0) - pivot["lat"]
    ).clip(lower=0)
    return pivot.sort_values("arrival_spread", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Customer delay risk via PERT variance propagation
# ─────────────────────────────────────────────────────────────────────────────
def customer_delay_risk(
    result: SolverResult,
    data: CVRPTWData,
    late_prob_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Estimate probability of late arrival at each customer using PERT statistics.

    μ_arrival is taken from the most-likely simulation.
    σ_arrival² is propagated cumulatively along the route.

    P(late) = P(T > LAT) = 1 - Φ((LAT - μ) / σ)

    Parameters
    ----------
    late_prob_threshold : customers with P(late) > threshold are flagged.
    """
    mu_matrix, var_matrix = compute_pert_statistics(data)
    records = []

    for route_idx, route in enumerate(result.routes):
        mu_arrivals  = _simulate_route_arrivals(
            route, mu_matrix, data.service_times
        )
        cum_vars = _propagate_variance(route, var_matrix, data.service_times)

        for pos, (node, mu_arr, cum_var) in enumerate(
            zip(route, mu_arrivals, cum_vars)
        ):
            if node == DEPOT_NODE:
                continue
            lat  = float(data.lat[node])
            sigma = np.sqrt(cum_var) if cum_var > 0 else 1e-6
            p_late = 1.0 - norm.cdf((lat - mu_arr) / sigma)
            records.append({
                "route":          route_idx + 1,
                "position":       pos,
                "node":           node,
                "mu_arrival":     round(mu_arr, 2),
                "sigma_arrival":  round(sigma, 2),
                "lat":            int(lat),
                "slack_min":      round(lat - mu_arr, 2),
                "p_late":         round(p_late, 4),
                "high_risk":      p_late > late_prob_threshold,
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("p_late", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Arc sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def arc_uncertainty_report(
    result: SolverResult,
    data: CVRPTWData,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    For arcs actually used in the solution, compute their uncertainty ratio
    and rank them by risk.
    """
    ratio = compute_arc_uncertainty_ratio(data)
    a_mat = data.time_matrices["optimistic"].astype(float)
    m_mat = data.time_matrices["mostlikely"].astype(float)
    b_mat = data.time_matrices["pessimistic"].astype(float)

    records = []
    for route_idx, route in enumerate(result.routes):
        for k in range(len(route) - 1):
            i, j = route[k], route[k + 1]
            records.append({
                "route":           route_idx + 1,
                "from":            i,
                "to":              j,
                "t_optimistic":    int(a_mat[i][j]),
                "t_mostlikely":    int(m_mat[i][j]),
                "t_pessimistic":   int(b_mat[i][j]),
                "uncertainty_ratio": round(float(ratio[i][j]), 3),
                "range_min":       int(b_mat[i][j] - a_mat[i][j]),
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("uncertainty_ratio", ascending=False).head(top_n)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Route stability score
# ─────────────────────────────────────────────────────────────────────────────
def route_stability_scores(
    result: SolverResult,
    data: CVRPTWData,
) -> pd.DataFrame:
    """
    Per-route stability score = 1 / (1 + mean_arrival_spread).
    Score closer to 1.0 → more stable under uncertainty.
    """
    spread_df = route_arrival_deviation(result, data)
    if spread_df.empty:
        return pd.DataFrame()

    route_stats = (
        spread_df.groupby("route")["arrival_spread"]
        .agg(["mean", "max", "sum"])
        .rename(columns={"mean": "mean_spread", "max": "max_spread", "sum": "total_spread"})
        .reset_index()
    )
    route_stats["stability_score"] = 1.0 / (1.0 + route_stats["mean_spread"])
    route_stats = route_stats.sort_values("stability_score")
    return route_stats


# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive robustness summary
# ─────────────────────────────────────────────────────────────────────────────
def full_robustness_report(
    result: SolverResult,
    data: CVRPTWData,
    late_prob_threshold: float = 0.10,
) -> Dict:
    """Return all robustness artefacts in a single dictionary."""
    return {
        "arrival_deviation":  route_arrival_deviation(result, data),
        "delay_risk":         customer_delay_risk(result, data, late_prob_threshold),
        "arc_uncertainty":    arc_uncertainty_report(result, data),
        "stability_scores":   route_stability_scores(result, data),
    }


if __name__ == "__main__":
    from src.data_loader import load_day
    from src.cvrptw_solver import solve

    data   = load_day(1)
    result = solve(data, scenario="mostlikely", time_limit_s=30)

    if result.is_solved():
        report = full_robustness_report(result, data)
        print("=== Delay Risk (top 10) ===")
        print(report["delay_risk"].head(10).to_string(index=False))
        print()
        print("=== Route Stability Scores ===")
        print(report["stability_scores"].to_string(index=False))
        print()
        print("=== Arc Uncertainty (top 10) ===")
        print(report["arc_uncertainty"].head(10).to_string(index=False))
    else:
        print(f"Solver returned: {result.status}")
