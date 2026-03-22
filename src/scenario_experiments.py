"""
scenario_experiments.py
-----------------------
Structured experiments measuring CVRPTW sensitivity to three operational levers:

  1. Demand scaling   – scale weights/volumes by a factor (simulate surge orders)
  2. Time-window tightening – reduce LAT uniformly (more demanding SLA)
  3. Fleet reduction  – decrease the number of available vehicles

Each experiment returns a DataFrame of KPIs for plotting and analysis.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_loader import CVRPTWData
from src.cvrptw_solver import (
    CVRPTWSolver,
    SolverResult,
    DEFAULT_NUM_VEHICLES,
    DEFAULT_MAX_WEIGHT_KG,
    DEFAULT_MAX_VOLUME_M3,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _patch_data(data: CVRPTWData, **overrides) -> CVRPTWData:
    """Return a shallow-copy of CVRPTWData with patched arrays."""
    patched = copy.copy(data)
    orders  = data.orders.copy()
    for k, v in overrides.items():
        if k == "orders":
            orders = v
        else:
            object.__setattr__(patched, k, v)
    object.__setattr__(patched, "orders", orders)
    # Recompute derived arrays
    patched.__post_init__()
    return patched


def _solve_scenario(
    data: CVRPTWData,
    scenario: str,
    num_vehicles: int,
    max_weight_kg: float,
    max_volume_m3: float,
    time_limit_s: int,
) -> SolverResult:
    solver = CVRPTWSolver(
        data=data,
        scenario=scenario,
        num_vehicles=num_vehicles,
        max_weight_kg=max_weight_kg,
        max_volume_m3=max_volume_m3,
        time_limit_s=time_limit_s,
    )
    return solver.solve()


def _result_to_row(
    r: SolverResult,
    **extra_fields,
) -> Dict:
    row = {
        "status":             r.status,
        "feasible":           int(r.is_solved()),
        "vehicles_used":      r.num_vehicles_used,
        "total_travel_min":   round(r.total_travel_time, 1),
        "total_dist_km":      round(r.total_distance_km, 1),
        "late_deliveries":    r.late_deliveries,
        "service_level":      round(
            1.0 - r.late_deliveries / max(1, r.num_vehicles_used * 10), 4
        ),
        "solve_time_s":       round(r.wall_time_s, 2),
    }
    row.update(extra_fields)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 – Demand Scaling
# ─────────────────────────────────────────────────────────────────────────────
def experiment_demand_scaling(
    data: CVRPTWData,
    scale_factors: Optional[List[float]] = None,
    scenario: str = "mostlikely",
    num_vehicles: int = DEFAULT_NUM_VEHICLES,
    max_weight_kg: float = DEFAULT_MAX_WEIGHT_KG,
    max_volume_m3: float = DEFAULT_MAX_VOLUME_M3,
    time_limit_s: int = 45,
) -> pd.DataFrame:
    """
    Increase demand by scaling weights and volumes.  Measures feasibility and
    cost degradation as orders grow.

    Parameters
    ----------
    scale_factors : e.g. [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    """
    if scale_factors is None:
        scale_factors = [0.75, 1.0, 1.25, 1.50, 1.75, 2.0]

    records = []
    for sf in scale_factors:
        print(f"  demand_scale={sf:.2f} …", end=" ", flush=True)
        orders_scaled = data.orders.copy()
        orders_scaled["WEIGHT"] = data.orders["WEIGHT"] * sf
        orders_scaled["VOLUME"] = data.orders["VOLUME"] * sf

        patched = _patch_data(data, orders=orders_scaled)
        r = _solve_scenario(patched, scenario, num_vehicles, max_weight_kg, max_volume_m3, time_limit_s)
        row = _result_to_row(r, demand_scale=sf)
        records.append(row)
        print(f"status={r.status}  travel={r.total_travel_time:.0f}min  late={r.late_deliveries}")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 – Time-Window Tightening
# ─────────────────────────────────────────────────────────────────────────────
def experiment_tw_tightening(
    data: CVRPTWData,
    lat_fractions: Optional[List[float]] = None,
    scenario: str = "mostlikely",
    num_vehicles: int = DEFAULT_NUM_VEHICLES,
    max_weight_kg: float = DEFAULT_MAX_WEIGHT_KG,
    max_volume_m3: float = DEFAULT_MAX_VOLUME_M3,
    time_limit_s: int = 45,
) -> pd.DataFrame:
    """
    Tighten time windows by reducing LAT as a fraction of its original value.
    Smaller fraction → stricter deadline.

    Parameters
    ----------
    lat_fractions : e.g. [1.0, 0.9, 0.8, 0.7, 0.6]
    """
    if lat_fractions is None:
        lat_fractions = [1.0, 0.90, 0.80, 0.70, 0.60]

    records = []
    for frac in lat_fractions:
        print(f"  lat_fraction={frac:.2f} …", end=" ", flush=True)
        orders_tw = data.orders.copy()
        orders_tw["LAT"] = (data.orders["LAT"] * frac).astype(int).clip(lower=orders_tw["EAT"] + 1)

        patched = _patch_data(data, orders=orders_tw)
        r = _solve_scenario(patched, scenario, num_vehicles, max_weight_kg, max_volume_m3, time_limit_s)
        row = _result_to_row(r, lat_fraction=frac)
        records.append(row)
        print(f"status={r.status}  travel={r.total_travel_time:.0f}min  late={r.late_deliveries}")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3 – Fleet Size Reduction
# ─────────────────────────────────────────────────────────────────────────────
def experiment_fleet_reduction(
    data: CVRPTWData,
    fleet_sizes: Optional[List[int]] = None,
    scenario: str = "mostlikely",
    max_weight_kg: float = DEFAULT_MAX_WEIGHT_KG,
    max_volume_m3: float = DEFAULT_MAX_VOLUME_M3,
    time_limit_s: int = 45,
) -> pd.DataFrame:
    """
    Reduce fleet size and measure impact on feasibility and travel cost.

    Parameters
    ----------
    fleet_sizes : e.g. [10, 9, 8, 7, 6, 5]
    """
    if fleet_sizes is None:
        fleet_sizes = [10, 9, 8, 7, 6, 5]

    records = []
    for k in fleet_sizes:
        print(f"  num_vehicles={k} …", end=" ", flush=True)
        r = _solve_scenario(data, scenario, k, max_weight_kg, max_volume_m3, time_limit_s)
        row = _result_to_row(r, num_vehicles_available=k)
        records.append(row)
        print(f"status={r.status}  travel={r.total_travel_time:.0f}min  late={r.late_deliveries}")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4 – Capacity Sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def experiment_capacity_sensitivity(
    data: CVRPTWData,
    weight_caps: Optional[List[float]] = None,
    scenario: str = "mostlikely",
    num_vehicles: int = DEFAULT_NUM_VEHICLES,
    max_volume_m3: float = DEFAULT_MAX_VOLUME_M3,
    time_limit_s: int = 45,
) -> pd.DataFrame:
    """
    Vary weight capacity per vehicle to find the effective lower bound.

    Parameters
    ----------
    weight_caps : list of weight capacities in kg, e.g. [600, 500, 400, 350, 300]
    """
    if weight_caps is None:
        weight_caps = [600, 500, 400, 350, 300]

    records = []
    for wc in weight_caps:
        print(f"  max_weight_kg={wc} …", end=" ", flush=True)
        r = _solve_scenario(data, scenario, num_vehicles, float(wc), max_volume_m3, time_limit_s)
        row = _result_to_row(r, max_weight_kg=wc)
        records.append(row)
        print(f"status={r.status}  travel={r.total_travel_time:.0f}min  late={r.late_deliveries}")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Run all experiments
# ─────────────────────────────────────────────────────────────────────────────
def run_all_experiments(
    data: CVRPTWData,
    scenario: str = "mostlikely",
    time_limit_s: int = 45,
) -> Dict[str, pd.DataFrame]:
    """Run all four experiments and return results keyed by experiment name."""
    print("=== Experiment 1: Demand Scaling ===")
    df_demand = experiment_demand_scaling(data, scenario=scenario, time_limit_s=time_limit_s)

    print("\n=== Experiment 2: Time-Window Tightening ===")
    df_tw = experiment_tw_tightening(data, scenario=scenario, time_limit_s=time_limit_s)

    print("\n=== Experiment 3: Fleet Reduction ===")
    df_fleet = experiment_fleet_reduction(data, scenario=scenario, time_limit_s=time_limit_s)

    print("\n=== Experiment 4: Capacity Sensitivity ===")
    df_cap = experiment_capacity_sensitivity(data, scenario=scenario, time_limit_s=time_limit_s)

    return {
        "demand_scaling":      df_demand,
        "tw_tightening":       df_tw,
        "fleet_reduction":     df_fleet,
        "capacity_sensitivity":df_cap,
    }


if __name__ == "__main__":
    from src.data_loader import load_day

    data = load_day(1)
    results = run_all_experiments(data, time_limit_s=30)
    for name, df in results.items():
        print(f"\n{name}:")
        print(df.to_string(index=False))
