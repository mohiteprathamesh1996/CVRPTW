"""
data_loader.py
--------------
Loads, validates, and structures all CVRPTW input data from the Zenodo dataset.

Data layout
-----------
extracted_data/
  orders/orders.xlsx                       - 78 customer nodes with demand/TW
  time_and_distance_matrices/day_{d}/      - per-day matrices (d = 1..9)
    distance_matrix_{d}.xlsx
    time_matrix_optimistic_{d}.xlsx
    time_matrix_mostlikely_{d}.xlsx
    time_matrix_pessimistic_{d}.xlsx
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "extracted_data"
ORDERS_PATH = DATA_DIR / "orders" / "orders.xlsx"
MATRICES_DIR = DATA_DIR / "time_and_distance_matrices"

DEPOT_NODE = 0
SCENARIOS = ("optimistic", "mostlikely", "pessimistic")
AVAILABLE_DAYS = list(range(1, 10))


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CVRPTWData:
    """Fully validated CVRPTW instance for a single day."""

    day: int
    orders: pd.DataFrame                          # rows = customers, indexed by NODE_ID
    distance_matrix: np.ndarray                   # (N+1) × (N+1)  km
    time_matrices: Dict[str, np.ndarray]          # scenario → (N+1)×(N+1) minutes
    node_ids: np.ndarray                          # [0, 1, …, N]  (0 = depot)

    # Derived convenience arrays (depot first, then customers)
    weights: np.ndarray = field(init=False)       # kg
    volumes: np.ndarray = field(init=False)       # m³
    service_times: np.ndarray = field(init=False) # minutes
    eat: np.ndarray = field(init=False)           # earliest arrival (minutes)
    lat: np.ndarray = field(init=False)           # latest arrival (minutes)

    def __post_init__(self) -> None:
        n = len(self.orders)
        self.weights       = np.zeros(n + 1);  self.weights[1:]       = self.orders["WEIGHT"].values
        self.volumes       = np.zeros(n + 1);  self.volumes[1:]       = self.orders["VOLUME"].values
        self.service_times = np.zeros(n + 1, dtype=int);  self.service_times[1:] = self.orders["SERVICE_TIME"].values.astype(int)
        self.eat           = np.zeros(n + 1, dtype=int);  self.eat[1:]           = self.orders["EAT"].values.astype(int)
        self.lat           = np.full(n + 1, 999999, dtype=int);  self.lat[1:]   = self.orders["LAT"].values.astype(int)
        # depot: open all day
        self.lat[0] = 999999

    # ── Convenience ──────────────────────────────────────────────────────────
    @property
    def num_customers(self) -> int:
        return len(self.orders)

    @property
    def num_nodes(self) -> int:
        return len(self.orders) + 1

    def time_matrix(self, scenario: str) -> np.ndarray:
        if scenario not in self.time_matrices:
            raise KeyError(f"Unknown scenario '{scenario}'. Choose from {SCENARIOS}.")
        return self.time_matrices[scenario]

    def summary(self) -> pd.DataFrame:
        rows = []
        for scen in SCENARIOS:
            tm = self.time_matrices[scen]
            rows.append({
                "scenario":        scen,
                "mean_travel_min": round(tm[tm > 0].mean(), 2),
                "max_travel_min":  int(tm.max()),
                "min_travel_min":  int(tm[tm > 0].min()),
            })
        df = pd.DataFrame(rows).set_index("scenario")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _validate_matrix(mat: pd.DataFrame, name: str, expected_size: int) -> None:
    if mat.shape != (expected_size, expected_size):
        raise ValueError(f"{name}: expected {expected_size}×{expected_size}, got {mat.shape}")
    if mat.isnull().any().any():
        missing = int(mat.isnull().sum().sum())
        warnings.warn(f"{name}: {missing} NaN values found — filling with 0.")
    if (mat.values < 0).any():
        raise ValueError(f"{name}: negative values detected.")


def _validate_orders(orders: pd.DataFrame) -> None:
    required = {"NODE_ID", "WEIGHT", "VOLUME", "SERVICE_TIME", "EAT", "LAT"}
    missing_cols = required - set(orders.columns)
    if missing_cols:
        raise ValueError(f"orders.xlsx missing columns: {missing_cols}")
    if orders["WEIGHT"].lt(0).any():
        raise ValueError("Negative weight values in orders.")
    if orders["VOLUME"].lt(0).any():
        raise ValueError("Negative volume values in orders.")
    if (orders["EAT"] > orders["LAT"]).any():
        bad = orders[orders["EAT"] > orders["LAT"]]["NODE_ID"].tolist()
        raise ValueError(f"EAT > LAT for nodes: {bad}")


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_orders(path: Optional[Path] = None) -> pd.DataFrame:
    """Load and validate customer orders."""
    p = path or ORDERS_PATH
    orders = pd.read_excel(p)
    _validate_orders(orders)
    orders = orders.sort_values("NODE_ID").reset_index(drop=True)
    return orders


def _load_matrix(path: Path, name: str, expected_size: int) -> np.ndarray:
    df = pd.read_excel(path, index_col=0)
    df = df.fillna(0)
    _validate_matrix(df, name, expected_size)
    return df.values.astype(float)


def load_day(day: int, orders: Optional[pd.DataFrame] = None) -> CVRPTWData:
    """
    Load all matrices for a given day and return a CVRPTWData instance.

    Parameters
    ----------
    day : int
        Day number (1–9).
    orders : pd.DataFrame, optional
        Pre-loaded orders table; loaded from disk if None.
    """
    if day not in AVAILABLE_DAYS:
        raise ValueError(f"day must be in {AVAILABLE_DAYS}, got {day}.")

    orders = orders if orders is not None else load_orders()
    n_nodes = len(orders) + 1          # +1 for depot (node 0)
    day_dir = MATRICES_DIR / f"day_{day}"

    dist_path = day_dir / f"distance_matrix_{day}.xlsx"
    if not dist_path.exists():
        raise FileNotFoundError(f"Distance matrix not found: {dist_path}")
    distance_matrix = _load_matrix(dist_path, f"distance_matrix_{day}", n_nodes)

    time_matrices: Dict[str, np.ndarray] = {}
    for scen in SCENARIOS:
        fname = f"time_matrix_{scen}_{day}.xlsx"
        tm_path = day_dir / fname
        if not tm_path.exists():
            raise FileNotFoundError(f"Time matrix not found: {tm_path}")
        time_matrices[scen] = _load_matrix(tm_path, fname, n_nodes)

    node_ids = np.arange(n_nodes, dtype=int)

    return CVRPTWData(
        day=day,
        orders=orders,
        distance_matrix=distance_matrix,
        time_matrices=time_matrices,
        node_ids=node_ids,
    )


def load_all_days(orders: Optional[pd.DataFrame] = None) -> Dict[int, CVRPTWData]:
    """Load data for all 9 days."""
    orders = orders or load_orders()
    return {day: load_day(day, orders) for day in AVAILABLE_DAYS}


# ─────────────────────────────────────────────────────────────────────────────
# Quick exploratory summary
# ─────────────────────────────────────────────────────────────────────────────
def orders_eda(orders: pd.DataFrame) -> None:
    """Print a compact exploratory summary of the orders dataset."""
    print(f"Customers        : {len(orders)}")
    print(f"Total weight     : {orders['WEIGHT'].sum():.1f} kg")
    print(f"Total volume     : {orders['VOLUME'].sum():.4f} m³")
    print(f"Weight range     : [{orders['WEIGHT'].min():.3f}, {orders['WEIGHT'].max():.3f}] kg")
    print(f"Volume range     : [{orders['VOLUME'].min():.5f}, {orders['VOLUME'].max():.5f}] m³")
    print(f"Service times    : {sorted(orders['SERVICE_TIME'].unique())} min")
    print(f"Time windows (LAT): {sorted(orders['LAT'].unique())} min")
    tw_counts = orders["LAT"].value_counts().sort_index()
    for lat, cnt in tw_counts.items():
        print(f"  LAT={lat:4d} min : {cnt} customers")


if __name__ == "__main__":
    orders = load_orders()
    orders_eda(orders)
    data = load_day(1, orders)
    print(f"\nDay 1 — {data.num_customers} customers, {data.num_nodes} nodes")
    print(data.summary())
