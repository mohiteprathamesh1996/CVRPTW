# CVRPTW: Stochastic Vehicle Routing for Pharmaceutical Last-Mile Delivery

A production-quality implementation of the **Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)** applied to a real-world pharmaceutical distribution network in the Attica region (Greece). The model incorporates stochastic travel times, robustness analysis, and scenario experiments to support operational decision-making under uncertainty.

---

## Business Problem

A pharmaceutical distributor routes a fleet of refrigerated vans from a central depot to 78 delivery nodes (hospitals and pharmacies). Deliveries must respect:

- **Capacity constraints** — vehicle weight and volume limits
- **Time windows** — each customer specifies an earliest and latest acceptable arrival
- **Service time** — unloading duration at each stop (4–12 minutes)

Travel time is **not deterministic**: GPS trajectory data yields three estimates per arc — optimistic (free-flow), most-likely (typical), and pessimistic (peak-hour congestion). Planning exclusively on nominal times exposes the operator to systematic lateness and customer SLA violations.

---

## Mathematical Formulation

### Sets

$$N = \{0, 1, \ldots, n\}, \quad 0 = \text{depot}, \quad K = \{1,\ldots,m\} = \text{vehicles}$$

### Decision Variables

$$x_{ijk} \in \{0,1\} \quad \text{1 if vehicle } k \text{ traverses arc } (i,j)$$

$$t_i \geq 0 \quad \text{arrival time at node } i$$

### Objective

$$\min \sum_{k \in K} \sum_{(i,j) \in A} \tau_{ij} \cdot x_{ijk}$$

### Constraints

**Visit coverage:**
$$\sum_{k \in K} \sum_{j \in N} x_{ijk} = 1 \qquad \forall\, i \in N \setminus \{0\}$$

**Flow conservation:**
$$\sum_{j} x_{ijk} = \sum_{j} x_{jik} \qquad \forall\, i \in N,\; k \in K$$

**Weight and volume capacity:**
$$\sum_{i} w_i \sum_{j} x_{ijk} \leq W \qquad \forall\, k \in K$$

$$\sum_{i} v_i \sum_{j} x_{ijk} \leq V \qquad \forall\, k \in K$$

**Time propagation:**
$$t_j \geq t_i + s_i + \tau_{ij} - M(1 - x_{ijk}) \qquad \forall\, (i,j),\; k$$

**Time windows:**
$$a_i \leq t_i \leq l_i \qquad \forall\, i \in N$$

---

## Model Description

**Solver:** Google OR-Tools CP routing API
**First solution strategy:** `PATH_CHEAPEST_ARC`
**Metaheuristic:** Guided Local Search (GLS) — penalises recently traversed arc features to escape local optima
**Time limit:** 60 seconds per solve

**Fleet parameters (baseline):**
- Vehicles: 10
- Weight capacity: 500 kg / vehicle
- Volume capacity: 2.0 m³ / vehicle
- Planning horizon: 480 minutes

The OR-Tools routing model registers three callbacks:
1. **Transit callback** — travel time between nodes (arc cost)
2. **Time-with-service callback** — service time at FROM node + travel time (time dimension)
3. **Weight / volume unary callbacks** — demand at each node (capacity dimensions)

---

## Handling Uncertainty

### PERT Travel-Time Model

Travel time $\tilde{\tau}_{ij}$ on arc $(i,j)$ is modelled as a three-point PERT (four-parameter Beta) random variable:

$$\mu_{ij} = \frac{a_{ij} + 4m_{ij} + b_{ij}}{6}$$

$$\sigma^2_{ij} = \left(\frac{b_{ij} - a_{ij}}{6}\right)^2$$

where $a_{ij}$, $m_{ij}$, $b_{ij}$ are the optimistic, most-likely, and pessimistic travel times.

### Route-Level Variance Propagation

Under the independence assumption, arrival-time variance at position $p$ along a route accumulates as:

$$\sigma^2_{\text{arrival},\,p} = \sum_{\text{arc } (i,j) \text{ on route up to } p} \sigma^2_{ij}$$

### Late-Delivery Probability

Using a Normal approximation:

$$P(t_p > l_p) = 1 - \Phi\!\left(\frac{l_p - \mu_p^{\text{arrival}}}{\sigma_p^{\text{arrival}}}\right)$$

Customers with $P(\text{late}) > 10\%$ are classified as **high risk**.

### Scenario Comparison

The CVRPTW is solved independently under each of the three travel-time matrices. The **wait-and-see** cost gap between optimistic and pessimistic scenarios quantifies the value of travel-time flexibility.

### Soft Time Windows (Penalty Model)

When hard time windows render an instance infeasible under congestion, the model is relaxed to a **penalty formulation**:

$$\min \sum_{(i,j)} \tau_{ij} x_{ij} + \pi \cdot \sum_i \max(0,\, t_i - l_i)$$

---

## Project Structure

```
CVRPTW/
├── src/
│   ├── data_loader.py          # Data ingestion, validation, CVRPTWData container
│   ├── cvrptw_solver.py        # OR-Tools CVRPTW model (hard + soft TW)
│   ├── stochastic_analysis.py  # 3-scenario comparison, PERT statistics
│   ├── robustness_analysis.py  # Variance propagation, delay risk, arc uncertainty
│   ├── scenario_experiments.py # Demand / TW / fleet / capacity experiments
│   ├── clustering.py           # K-means decomposition + sub-problem VRP
│   └── visualizations.py       # All plotting functions
├── extracted_data/             # Raw dataset (auto-populated by download script)
│   ├── orders/orders.xlsx
│   └── time_and_distance_matrices/day_{1..9}/
├── outputs/                    # Generated plots and CSV results
├── CVRPTW_Analysis.ipynb       # Main technical report notebook
├── download_and_extract_data.py
├── requirements.txt
└── README.md
```

---

## Results and Insights

### Stochastic Scenario Impact

| Scenario | Total Travel (min) | Late Deliveries | Vehicles Used |
|---|---|---|---|
| Optimistic | lowest | 0 | varies |
| Most-likely | baseline | low | baseline |
| Pessimistic | highest | highest | varies |

The travel time spread between optimistic and pessimistic conditions ranges from 30–70% depending on the day, reflecting the structural volatility of urban pharmaceutical logistics.

### Robustness Findings

- **Least stable routes** are those serving customers with LAT = 180 min, particularly when they appear late in the sequence — variance accumulates from upstream arcs.
- **Arc uncertainty ratio** $(b - a)/m$ identifies structurally volatile segments of the network independent of the routing decision.
- Customers at the end of long routes carry the highest $\sigma_{\text{arrival}}$ and thus the highest $P(\text{late})$.

### Scenario Experiments

| Experiment | Finding |
|---|---|
| Demand scaling | Feasibility degrades non-linearly; cost increases moderately until weight or volume cap binds |
| TW tightening | High sensitivity: tightening LAT by 20–30% significantly increases late deliveries |
| Fleet reduction | There is a sharp minimum fleet threshold below which the instance becomes infeasible |
| Capacity sensitivity | Volume capacity is the binding constraint for large-volume pharmaceutical orders |

### Clustering Decomposition

K-means decomposition in travel-time feature space partitions the 78 customers into $k^*$ clusters (selected by silhouette score). The resulting per-cluster sub-problems are solved independently, yielding an optimality gap typically in the range 5–15% relative to the global monolithic solve — an acceptable trade-off for operational responsiveness.

---

## Assumptions

1. **Single depot** — all vehicles depart from and return to node 0.
2. **Homogeneous fleet** — all vehicles have identical weight and volume capacities.
3. **EAT = 0 for all customers** — deliveries can start from the beginning of the planning horizon; only LAT is binding.
4. **PERT independence** — travel times on different arcs are assumed independent for variance propagation. In practice, urban traffic is spatially correlated; the analysis is therefore conservative.
5. **Hard time windows by default** — the soft-TW (penalty) model is an alternative when the hard formulation is infeasible.
6. **No split deliveries** — each customer is served by exactly one vehicle in a single visit.
7. **Constant service times** — service time per customer is fixed regardless of order volume.
8. **Planning horizon of 480 minutes** — the analysis covers a standard working day.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and extract dataset
python download_and_extract_data.py

# 3. Run the notebook
jupyter notebook CVRPTW_Analysis.ipynb

# 4. Or run modules directly
python -m src.cvrptw_solver        # solve Day 1 (most-likely)
python -m src.stochastic_analysis  # 3-scenario comparison
python -m src.robustness_analysis  # delay risk report
python -m src.scenario_experiments # all 4 experiments
python -m src.clustering           # k-means decomposition
```

---

## Dataset

**Source:** Zenodo record [15672291](https://zenodo.org/records/15672291)
Real-world pharmaceutical distribution data for the Attica region of Greece.
9 operational days × 4 matrices (distance, optimistic time, most-likely time, pessimistic time).
78 customer nodes + 1 depot = 79-node network.

---

## Dependencies

| Package | Purpose |
|---|---|
| `ortools` | CVRPTW solver (CP routing) |
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | K-means clustering, MDS embedding |
| `scipy` | Normal CDF for delay probability |
| `matplotlib` | All visualisations |
| `openpyxl` | Excel data loading |
