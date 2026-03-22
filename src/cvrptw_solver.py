"""
cvrptw_solver.py
----------------
Capacitated Vehicle Routing Problem with Time Windows (CVRPTW) solver
using Google OR-Tools CP-SAT + Routing API.

Mathematical formulation
------------------------
Sets
    N = {0,1,...,n}   nodes (0 = depot, 1..n = customers)
    K = {1,...,m}     vehicles
    A = {(i,j) : i≠j}  arcs

Decision variables
    x_{ijk} ∈ {0,1}  vehicle k traverses arc (i,j)
    t_i     ≥ 0      arrival time at node i

Objective
    min  Σ_{k∈K} Σ_{(i,j)∈A}  c_{ij} · x_{ijk}

Constraints
    Σ_{k} Σ_{j} x_{ijk} = 1            ∀ i ∈ N/{0}   (every customer visited once)
    Σ_{j} x_{0jk} = 1                  ∀ k ∈ K        (each vehicle leaves depot)
    Σ_{j} x_{jik} = Σ_{j} x_{ijk}     ∀ i,k          (flow conservation)
    Σ_{i,j} w_i · x_{ijk} ≤ W_max     ∀ k            (weight capacity)
    Σ_{i,j} v_i · x_{ijk} ≤ V_max     ∀ k            (volume capacity)
    t_j ≥ t_i + s_i + τ_{ij}           ∀ (i,j),k     (time propagation)
    a_i ≤ t_i ≤ l_i                    ∀ i ∈ N       (time windows)

Arc cost  c_{ij} = travel time τ_{ij}  (minutes)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.data_loader import CVRPTWData, DEPOT_NODE, SCENARIOS

# ─────────────────────────────────────────────────────────────────────────────
# Fleet configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_NUM_VEHICLES   = 10
DEFAULT_MAX_WEIGHT_KG  = 500.0   # kg per vehicle
DEFAULT_MAX_VOLUME_M3  = 2.0     # m³ per vehicle
TIME_HORIZON_MIN       = 480     # planning horizon (minutes)

# Integer scaling factors for OR-Tools (which requires integer callbacks)
WEIGHT_SCALE = 1000              # kg → grams equivalent
VOLUME_SCALE = 100_000           # m³ → integer (5 decimal precision)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SolverResult:
    status: str                             # 'OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'NOT_SOLVED'
    scenario: str
    day: int
    num_vehicles_used: int
    total_travel_time: float                # minutes
    total_distance_km: float
    routes: List[List[int]]                 # list of node sequences (depot excluded from display)
    arrival_times: List[List[float]]        # arrival time at each node per route
    route_loads_weight: List[float]         # kg per route
    route_loads_volume: List[float]         # m³ per route
    late_deliveries: int                    # customers arriving after LAT
    wall_time_s: float
    vehicle_params: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Day {self.day} | Scenario: {self.scenario} | Status: {self.status}",
            f"  Vehicles used    : {self.num_vehicles_used}",
            f"  Total travel time: {self.total_travel_time:.1f} min",
            f"  Total distance   : {self.total_distance_km:.1f} km",
            f"  Late deliveries  : {self.late_deliveries}",
            f"  Solve time       : {self.wall_time_s:.2f} s",
        ]
        for i, route in enumerate(self.routes):
            wt = self.route_loads_weight[i]
            vl = self.route_loads_volume[i]
            lines.append(f"  Route {i+1:2d}: {route}  [W={wt:.1f}kg, V={vl:.3f}m³]")
        return "\n".join(lines)

    def is_solved(self) -> bool:
        return self.status in ("OPTIMAL", "FEASIBLE")

    def to_dict(self) -> Dict:
        return {
            "day":                self.day,
            "scenario":           self.scenario,
            "status":             self.status,
            "num_vehicles_used":  self.num_vehicles_used,
            "total_travel_time":  self.total_travel_time,
            "total_distance_km":  self.total_distance_km,
            "late_deliveries":    self.late_deliveries,
            "wall_time_s":        self.wall_time_s,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Core solver
# ─────────────────────────────────────────────────────────────────────────────
class CVRPTWSolver:
    """
    Wraps Google OR-Tools RoutingModel for CVRPTW.

    Parameters
    ----------
    data          : CVRPTWData instance
    scenario      : one of 'optimistic', 'mostlikely', 'pessimistic'
    num_vehicles  : fleet size
    max_weight_kg : weight capacity per vehicle (kg)
    max_volume_m3 : volume capacity per vehicle (m³)
    time_limit_s  : solver wall-clock limit
    penalty       : soft time-window violation penalty (0 = hard)
    """

    def __init__(
        self,
        data: CVRPTWData,
        scenario: str = "mostlikely",
        num_vehicles: int = DEFAULT_NUM_VEHICLES,
        max_weight_kg: float = DEFAULT_MAX_WEIGHT_KG,
        max_volume_m3: float = DEFAULT_MAX_VOLUME_M3,
        time_limit_s: int = 60,
        penalty: int = 0,
    ) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        self.data          = data
        self.scenario      = scenario
        self.num_vehicles  = num_vehicles
        self.max_weight_kg = max_weight_kg
        self.max_volume_m3 = max_volume_m3
        self.time_limit_s  = time_limit_s
        self.penalty       = penalty

        self._time_matrix  = data.time_matrix(scenario).astype(int)
        self._dist_matrix  = (data.distance_matrix * 10).astype(int)  # ×10 → integer

    # ── Build and solve ───────────────────────────────────────────────────────
    def solve(self) -> SolverResult:
        data      = self.data
        n_nodes   = data.num_nodes   # includes depot at index 0

        manager = pywrapcp.RoutingIndexManager(n_nodes, self.num_vehicles, DEPOT_NODE)
        routing = pywrapcp.RoutingModel(manager)

        # ── Transit callbacks ─────────────────────────────────────────────────
        # Time callback (travel time only — service time added separately)
        time_matrix_int = self._time_matrix

        def time_callback(from_idx, to_idx):
            i = manager.IndexToNode(from_idx)
            j = manager.IndexToNode(to_idx)
            return int(time_matrix_int[i][j])

        transit_cb_idx = routing.RegisterTransitCallback(time_callback)

        # Arc cost = travel time
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

        # Time callback with service time at FROM node (for time dimension)
        service_times = data.service_times  # integer minutes

        def time_with_service_callback(from_idx, to_idx):
            i = manager.IndexToNode(from_idx)
            j = manager.IndexToNode(to_idx)
            return int(service_times[i]) + int(time_matrix_int[i][j])

        ts_cb_idx = routing.RegisterTransitCallback(time_with_service_callback)

        # ── Time dimension ────────────────────────────────────────────────────
        routing.AddDimension(
            ts_cb_idx,
            slack_max=TIME_HORIZON_MIN,     # max waiting allowed at a node
            capacity=TIME_HORIZON_MIN,      # overall time horizon
            fix_start_cumul_to_zero=True,
            name="Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")

        for node in range(1, n_nodes):           # skip depot
            idx = manager.NodeToIndex(node)
            eat = int(data.eat[node])
            lat = int(data.lat[node])
            if self.penalty > 0:
                time_dim.SetCumulVarSoftUpperBound(idx, lat, self.penalty)
            else:
                time_dim.CumulVar(idx).SetRange(eat, lat)

        # ── Weight dimension ──────────────────────────────────────────────────
        weight_scaled = (data.weights * WEIGHT_SCALE).astype(int)
        max_w_int     = int(self.max_weight_kg * WEIGHT_SCALE)

        def weight_callback(from_idx):
            node = manager.IndexToNode(from_idx)
            return int(weight_scaled[node])

        wt_cb_idx = routing.RegisterUnaryTransitCallback(weight_callback)
        routing.AddDimensionWithVehicleCapacity(
            wt_cb_idx,
            0,
            [max_w_int] * self.num_vehicles,
            True,
            "Weight",
        )

        # ── Volume dimension ──────────────────────────────────────────────────
        volume_scaled = (data.volumes * VOLUME_SCALE).astype(int)
        max_v_int     = int(self.max_volume_m3 * VOLUME_SCALE)

        def volume_callback(from_idx):
            node = manager.IndexToNode(from_idx)
            return int(volume_scaled[node])

        vol_cb_idx = routing.RegisterUnaryTransitCallback(volume_callback)
        routing.AddDimensionWithVehicleCapacity(
            vol_cb_idx,
            0,
            [max_v_int] * self.num_vehicles,
            True,
            "Volume",
        )

        # ── Search parameters ─────────────────────────────────────────────────
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.seconds = self.time_limit_s
        search_params.log_search = False

        # ── Solve ─────────────────────────────────────────────────────────────
        import time
        t0 = time.perf_counter()
        solution = routing.SolveWithParameters(search_params)
        wall_time = time.perf_counter() - t0

        return self._extract_solution(
            routing, manager, solution, data, wall_time
        )

    # ── Solution extraction ───────────────────────────────────────────────────
    def _extract_solution(
        self,
        routing: pywrapcp.RoutingModel,
        manager: pywrapcp.RoutingIndexManager,
        solution,
        data: CVRPTWData,
        wall_time: float,
    ) -> SolverResult:

        status_map = {
            0: "NOT_SOLVED",
            1: "OPTIMAL",
            2: "FEASIBLE",
            3: "INFEASIBLE",
            4: "NOT_SOLVED",
        }
        status = status_map.get(routing.status(), "UNKNOWN")

        if solution is None or status in ("INFEASIBLE", "NOT_SOLVED"):
            return SolverResult(
                status=status,
                scenario=self.scenario,
                day=data.day,
                num_vehicles_used=0,
                total_travel_time=0.0,
                total_distance_km=0.0,
                routes=[],
                arrival_times=[],
                route_loads_weight=[],
                route_loads_volume=[],
                late_deliveries=0,
                wall_time_s=wall_time,
                vehicle_params={
                    "num_vehicles": self.num_vehicles,
                    "max_weight_kg": self.max_weight_kg,
                    "max_volume_m3": self.max_volume_m3,
                },
            )

        time_dim = routing.GetDimensionOrDie("Time")
        routes: List[List[int]]         = []
        arrivals: List[List[float]]     = []
        loads_wt: List[float]           = []
        loads_vl: List[float]           = []
        total_travel_time               = 0.0
        total_distance                  = 0.0
        late_count                      = 0

        for v in range(self.num_vehicles):
            idx = routing.Start(v)
            route_nodes: List[int] = []
            route_arr:   List[float] = []
            w_load = v_load = 0.0

            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                arr  = solution.Value(time_dim.CumulVar(idx))
                route_nodes.append(node)
                route_arr.append(float(arr))
                if node != DEPOT_NODE:
                    w_load += data.weights[node]
                    v_load += data.volumes[node]
                    if arr > data.lat[node]:
                        late_count += 1
                prev_idx = idx
                idx = solution.Value(routing.NextVar(idx))
                if not routing.IsEnd(idx):
                    i = manager.IndexToNode(prev_idx)
                    j = manager.IndexToNode(idx)
                    total_travel_time += self._time_matrix[i][j]
                    total_distance    += self.data.distance_matrix[i][j]

            if len(route_nodes) > 1:      # non-empty route (depot + customers)
                # append return-to-depot leg
                last_node = manager.IndexToNode(prev_idx)
                total_travel_time += self._time_matrix[last_node][DEPOT_NODE]
                total_distance    += self.data.distance_matrix[last_node][DEPOT_NODE]
                routes.append(route_nodes)
                arrivals.append(route_arr)
                loads_wt.append(w_load)
                loads_vl.append(v_load)

        return SolverResult(
            status=status,
            scenario=self.scenario,
            day=data.day,
            num_vehicles_used=len(routes),
            total_travel_time=total_travel_time,
            total_distance_km=total_distance,
            routes=routes,
            arrival_times=arrivals,
            route_loads_weight=loads_wt,
            route_loads_volume=loads_vl,
            late_deliveries=late_count,
            wall_time_s=wall_time,
            vehicle_params={
                "num_vehicles": self.num_vehicles,
                "max_weight_kg": self.max_weight_kg,
                "max_volume_m3": self.max_volume_m3,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Penalty-based soft time-window variant
# ─────────────────────────────────────────────────────────────────────────────
def solve_with_penalty(
    data: CVRPTWData,
    scenario: str = "mostlikely",
    penalty: int = 1000,
    **kwargs,
) -> SolverResult:
    """Solve CVRPTW with soft time-window violations (penalty-based)."""
    solver = CVRPTWSolver(data, scenario=scenario, penalty=penalty, **kwargs)
    return solver.solve()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────
def solve(
    data: CVRPTWData,
    scenario: str = "mostlikely",
    num_vehicles: int = DEFAULT_NUM_VEHICLES,
    max_weight_kg: float = DEFAULT_MAX_WEIGHT_KG,
    max_volume_m3: float = DEFAULT_MAX_VOLUME_M3,
    time_limit_s: int = 60,
) -> SolverResult:
    """Solve CVRPTW and return a SolverResult."""
    solver = CVRPTWSolver(
        data,
        scenario=scenario,
        num_vehicles=num_vehicles,
        max_weight_kg=max_weight_kg,
        max_volume_m3=max_volume_m3,
        time_limit_s=time_limit_s,
    )
    return solver.solve()


if __name__ == "__main__":
    from src.data_loader import load_day
    data = load_day(1)
    result = solve(data, scenario="mostlikely", time_limit_s=30)
    print(result)
