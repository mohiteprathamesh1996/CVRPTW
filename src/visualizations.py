"""
visualizations.py
-----------------
All visualisation functions for the CVRPTW project.

Functions
---------
plot_routes               – Network-style route graph
plot_time_matrix_heatmap  – Travel-time heatmap (scenario comparison)
plot_scenario_kpis        – Bar chart comparing scenario KPIs
plot_arrival_vs_window    – Gantt-style arrival times vs time windows
plot_delay_risk           – Customer delay risk scatter
plot_stability_scores     – Route stability bar chart
plot_experiment           – Generic experiment result line plots
plot_silhouette           – Silhouette score vs k
plot_cluster_assignment   – Customer clusters colour-coded
plot_pert_distribution    – PERT distribution for a specific arc
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # headless backend for notebook / script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from src.data_loader import CVRPTWData, DEPOT_NODE, SCENARIOS
from src.cvrptw_solver import SolverResult

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0",
           "#FF9800", "#00BCD4", "#E91E63", "#607D8B",
           "#795548", "#009688"]

SCENARIO_COLORS = {
    "optimistic":  "#4CAF50",
    "mostlikely":  "#2196F3",
    "pessimistic": "#FF5722",
}


# ─────────────────────────────────────────────────────────────────────────────
# Route network plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_routes(
    result: SolverResult,
    data: CVRPTWData,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Draw routes as a network diagram using travel-time as a proxy for
    spatial layout (MDS embedding).
    """
    from sklearn.manifold import MDS

    tm  = data.time_matrices["mostlikely"].astype(float)
    n   = data.num_nodes
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
    sym = (tm + tm.T) / 2.0
    pos = mds.fit_transform(sym)   # (N, 2) layout coordinates

    fig, ax = plt.subplots(figsize=figsize)

    # Draw routes
    for i, route in enumerate(result.routes):
        color = PALETTE[i % len(PALETTE)]
        coords = pos[route]
        ax.plot(coords[:, 0], coords[:, 1], "-o",
                color=color, linewidth=1.8, markersize=5,
                label=f"Route {i+1}  (W={result.route_loads_weight[i]:.0f}kg)", zorder=2)
        # Arrows for direction
        for k in range(len(route) - 1):
            dx = coords[k+1, 0] - coords[k, 0]
            dy = coords[k+1, 1] - coords[k, 1]
            ax.annotate("", xy=(coords[k+1, 0], coords[k+1, 1]),
                        xytext=(coords[k, 0], coords[k, 1]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                        zorder=3)

    # Highlight depot
    ax.scatter(*pos[0], s=300, color="black", zorder=5, marker="s", label="Depot")
    ax.annotate(" Depot", pos[0], fontsize=9, fontweight="bold")

    # Label all visited nodes
    visited = {n for route in result.routes for n in route if n != DEPOT_NODE}
    for node in visited:
        ax.annotate(str(node), pos[node], fontsize=6, ha="center", va="bottom", alpha=0.7)

    ax.set_title(
        title or f"Day {result.day} | {result.scenario.capitalize()} | "
                 f"{result.num_vehicles_used} routes | {result.total_travel_time:.0f} min",
        fontsize=13, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Travel-time heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_time_matrix_heatmap(
    data: CVRPTWData,
    scenario: str = "mostlikely",
    max_nodes: int = 30,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of a travel-time sub-matrix (first max_nodes nodes)."""
    import matplotlib.colors as mcolors

    tm = data.time_matrices[scenario][:max_nodes, :max_nodes]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(tm, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Travel time (min)")
    ax.set_title(f"Travel-time matrix — {scenario.capitalize()} scenario (first {max_nodes} nodes)",
                 fontsize=12)
    ax.set_xlabel("To node")
    ax.set_ylabel("From node")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Scenario KPI bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_scenario_kpis(
    kpi_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing KPIs across optimistic / most-likely / pessimistic."""
    metrics = ["Total Travel (min)", "Late Deliveries", "Vehicles Used"]
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)

    for ax, metric in zip(axes, metrics):
        if metric not in kpi_df.columns:
            ax.set_visible(False)
            continue
        vals   = kpi_df[metric].values
        labels = kpi_df.index.tolist()
        colors = [SCENARIO_COLORS.get(s.lower(), "#888") for s in labels]
        bars   = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(vals),
                    f"{v:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Scenario KPI Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Arrival vs time-window Gantt
# ─────────────────────────────────────────────────────────────────────────────
def plot_arrival_vs_window(
    result: SolverResult,
    data: CVRPTWData,
    max_routes: int = 5,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Gantt-style chart: time window bars with arrival markers per route."""
    n_routes = min(max_routes, len(result.routes))
    fig, axes = plt.subplots(n_routes, 1, figsize=(figsize[0], figsize[1] * n_routes // 2 + 2),
                             sharex=True)
    if n_routes == 1:
        axes = [axes]

    for ax, route_idx in zip(axes, range(n_routes)):
        route    = result.routes[route_idx]
        arrivals = result.arrival_times[route_idx]
        customers = [(n, a) for n, a in zip(route, arrivals) if n != DEPOT_NODE]
        if not customers:
            continue

        nodes, arrs = zip(*customers)
        lats  = [data.lat[n]  for n in nodes]
        eats  = [data.eat[n]  for n in nodes]
        y_pos = list(range(len(nodes)))

        for y, (eat, lat) in enumerate(zip(eats, lats)):
            ax.barh(y, lat - eat, left=eat, height=0.5,
                    color="#B3E5FC", edgecolor="#0288D1", linewidth=0.8, label="Time window" if y == 0 else "")

        late_mask = [a > l for a, l in zip(arrs, lats)]
        colors_arr = ["#E53935" if late else "#1B5E20" for late in late_mask]
        ax.scatter(arrs, y_pos, c=colors_arr, s=80, zorder=5,
                   label="Arrival (late)" if any(late_mask) else "Arrival (on-time)")

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Node {n}" for n in nodes], fontsize=8)
        ax.set_title(f"Route {route_idx+1}  —  W={result.route_loads_weight[route_idx]:.0f}kg",
                     fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Time (minutes)", fontsize=10)
    fig.suptitle(f"Arrival Times vs Time Windows — {result.scenario.capitalize()}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Delay risk scatter
# ─────────────────────────────────────────────────────────────────────────────
def plot_delay_risk(
    risk_df: pd.DataFrame,
    threshold: float = 0.10,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of P(late) vs slack for each customer."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: P(late) per customer
    ax = axes[0]
    colors = ["#E53935" if h else "#43A047" for h in risk_df["high_risk"]]
    ax.bar(risk_df["node"].astype(str), risk_df["p_late"], color=colors, width=0.8)
    ax.axhline(threshold, linestyle="--", color="gray", linewidth=1, label=f"Threshold ({threshold:.0%})")
    ax.set_xlabel("Customer Node")
    ax.set_ylabel("P(late)")
    ax.set_title("Late-Arrival Probability per Customer", fontsize=11)
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)

    # Right: slack vs sigma scatter
    ax2 = axes[1]
    sc = ax2.scatter(
        risk_df["sigma_arrival"],
        risk_df["slack_min"],
        c=risk_df["p_late"],
        cmap="RdYlGn_r",
        s=60, edgecolors="gray", linewidths=0.4
    )
    plt.colorbar(sc, ax=ax2, label="P(late)")
    ax2.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax2.set_xlabel("σ_arrival (min)")
    ax2.set_ylabel("Slack (LAT − μ_arrival) min")
    ax2.set_title("Risk Map: Slack vs Uncertainty", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Customer Delay Risk Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Route stability bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_stability_scores(
    stability_df: pd.DataFrame,
    figsize: Tuple[int, int] = (8, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart of route stability scores."""
    fig, ax = plt.subplots(figsize=figsize)
    labels  = [f"Route {r}" for r in stability_df["route"]]
    scores  = stability_df["stability_score"].values
    colors  = cm.RdYlGn(scores)

    bars = ax.barh(labels, scores, color=colors, edgecolor="white")
    ax.axvline(0.8, linestyle="--", color="gray", linewidth=1, label="Stability threshold (0.8)")
    for bar, s in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{s:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Stability Score (higher = more robust)")
    ax.set_title("Route Stability Under Travel-Time Uncertainty", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Experiment result line plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_experiment(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    x_label: str,
    title: str,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Generic line plot for scenario experiment results."""
    fig, axes = plt.subplots(1, len(y_cols), figsize=figsize)
    if len(y_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, y_cols):
        feasible = df[df["feasible"] == 1] if "feasible" in df.columns else df
        infeas   = df[df["feasible"] == 0] if "feasible" in df.columns else pd.DataFrame()

        ax.plot(feasible[x_col], feasible[col], "o-", color="#2196F3", linewidth=2, markersize=7)
        if not infeas.empty and col in infeas.columns:
            ax.scatter(infeas[x_col], [0] * len(infeas), marker="x", color="#E53935", s=100,
                       label="Infeasible", zorder=5)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(col.replace("_", " ").title(), fontsize=10)
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        if not infeas.empty:
            ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Silhouette score vs k
# ─────────────────────────────────────────────────────────────────────────────
def plot_silhouette(
    sil_df: pd.DataFrame,
    best_k: int,
    figsize: Tuple[int, int] = (7, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Line chart of silhouette score vs number of clusters."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(sil_df["k"], sil_df["silhouette"], "o-", color="#2196F3", linewidth=2)
    ax.axvline(best_k, linestyle="--", color="#FF5722", linewidth=1.5, label=f"Best k = {best_k}")
    ax.set_xlabel("Number of clusters (k)", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("K-Means: Silhouette Score vs k", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Cluster assignment (MDS layout)
# ─────────────────────────────────────────────────────────────────────────────
def plot_cluster_assignment(
    data: CVRPTWData,
    cluster_df: pd.DataFrame,
    figsize: Tuple[int, int] = (9, 7),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of customers in MDS space, coloured by cluster."""
    from sklearn.manifold import MDS
    from src.clustering import extract_node_features
    from sklearn.preprocessing import StandardScaler

    X  = extract_node_features(data)
    Xs = StandardScaler().fit_transform(X)
    mds = MDS(n_components=2, random_state=42, normalized_stress="auto")
    pos = mds.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=figsize)
    clusters = sorted(cluster_df["cluster"].unique())
    for c in clusters:
        mask  = cluster_df["cluster"] == c
        nodes = cluster_df.loc[mask, "node_id"].values - 1   # 0-indexed for pos
        ax.scatter(pos[nodes, 0], pos[nodes, 1],
                   color=PALETTE[c % len(PALETTE)], s=80, label=f"Cluster {c}",
                   edgecolors="white", linewidths=0.5, zorder=3)

    for i, row in cluster_df.iterrows():
        node = row["node_id"] - 1
        ax.annotate(str(row["node_id"]), (pos[node, 0], pos[node, 1]),
                    fontsize=5, ha="center", va="bottom", alpha=0.7)

    ax.set_title("Customer Clusters (MDS layout, time-space features)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PERT distribution for a specific arc
# ─────────────────────────────────────────────────────────────────────────────
def plot_pert_distribution(
    data: CVRPTWData,
    from_node: int,
    to_node: int,
    figsize: Tuple[int, int] = (8, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the PERT (four-parameter Beta) distribution for a single arc's travel time.
    """
    a = data.time_matrices["optimistic"][from_node][to_node]
    m = data.time_matrices["mostlikely"][from_node][to_node]
    b = data.time_matrices["pessimistic"][from_node][to_node]

    if b == a:
        return plt.figure()          # degenerate arc, skip

    mu  = (a + 4 * m + b) / 6.0
    var = ((b - a) / 6.0) ** 2
    # PERT shape parameters (scaled Beta on [a, b])
    lam = 4.0
    alpha1 = 1 + lam * (m - a) / (b - a)
    alpha2 = 1 + lam * (b - m) / (b - a)

    x  = np.linspace(a, b, 300)
    # Convert to standard [0,1] Beta
    x_std = (x - a) / (b - a)
    y  = beta_dist.pdf(x_std, alpha1, alpha2) / (b - a)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, color="#2196F3", linewidth=2, label="PERT density")
    ax.fill_between(x, y, alpha=0.2, color="#2196F3")
    ax.axvline(a,  linestyle="--", color="#4CAF50", linewidth=1.2, label=f"Optimistic ({a:.0f})")
    ax.axvline(m,  linestyle="-",  color="#FF9800", linewidth=1.5, label=f"Most-likely ({m:.0f})")
    ax.axvline(b,  linestyle="--", color="#F44336", linewidth=1.2, label=f"Pessimistic ({b:.0f})")
    ax.axvline(mu, linestyle=":",  color="black",   linewidth=1.2, label=f"PERT mean ({mu:.1f})")
    ax.set_xlabel("Travel time (min)")
    ax.set_ylabel("Density")
    ax.set_title(f"PERT Travel-Time Distribution — Arc ({from_node} → {to_node})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# KPI dashboard
# ─────────────────────────────────────────────────────────────────────────────
def plot_kpi_dashboard(
    results: Dict[str, SolverResult],
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Multi-panel KPI dashboard: travel time, vehicles, late deliveries, and
    per-route load utilisation.
    """
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    solved_scens = [s for s in SCENARIOS if s in results and results[s].is_solved()]
    colors = [SCENARIO_COLORS[s] for s in solved_scens]

    # 1. Total travel time
    ax1 = fig.add_subplot(gs[0, 0])
    vals = [results[s].total_travel_time for s in solved_scens]
    bars = ax1.bar([s.capitalize() for s in solved_scens], vals, color=colors, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01, f"{v:.0f}", ha="center", fontsize=9)
    ax1.set_title("Total Travel Time (min)", fontsize=11)
    ax1.spines[["top","right"]].set_visible(False)

    # 2. Vehicles used
    ax2 = fig.add_subplot(gs[0, 1])
    vals2 = [results[s].num_vehicles_used for s in solved_scens]
    bars2 = ax2.bar([s.capitalize() for s in solved_scens], vals2, color=colors, edgecolor="white")
    for bar, v in zip(bars2, vals2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01, str(v), ha="center", fontsize=9)
    ax2.set_title("Vehicles Used", fontsize=11)
    ax2.spines[["top","right"]].set_visible(False)

    # 3. Late deliveries
    ax3 = fig.add_subplot(gs[0, 2])
    vals3 = [results[s].late_deliveries for s in solved_scens]
    bars3 = ax3.bar([s.capitalize() for s in solved_scens], vals3, color=colors, edgecolor="white")
    for bar, v in zip(bars3, vals3):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01, str(v), ha="center", fontsize=9)
    ax3.set_title("Late Deliveries", fontsize=11)
    ax3.spines[["top","right"]].set_visible(False)

    # 4. Route weight utilisation
    ax4 = fig.add_subplot(gs[1, :2])
    ml_result = results.get("mostlikely")
    if ml_result and ml_result.is_solved():
        wt_util = [w / DEFAULT_MAX_WEIGHT_KG for w in ml_result.route_loads_weight]
        from src.cvrptw_solver import DEFAULT_MAX_WEIGHT_KG
        route_labels = [f"R{i+1}" for i in range(len(wt_util))]
        bar_colors   = ["#E53935" if u > 0.9 else "#43A047" if u < 0.5 else "#FF9800" for u in wt_util]
        ax4.bar(route_labels, [u * 100 for u in wt_util], color=bar_colors, edgecolor="white")
        ax4.axhline(80, linestyle="--", color="gray", linewidth=1, label="80% threshold")
        ax4.set_ylabel("Weight utilisation (%)")
        ax4.set_title("Route Weight Utilisation (Most-Likely)", fontsize=11)
        ax4.set_ylim(0, 115)
        ax4.legend(fontsize=9)
        ax4.spines[["top","right"]].set_visible(False)

    # 5. Travel time comparison bar
    ax5 = fig.add_subplot(gs[1, 2])
    if len(solved_scens) >= 2:
        base = results["mostlikely"].total_travel_time if "mostlikely" in results else vals[0]
        deltas = [(results[s].total_travel_time - base) for s in solved_scens]
        c_delta = ["#43A047" if d <= 0 else "#E53935" for d in deltas]
        ax5.bar([s.capitalize() for s in solved_scens], deltas, color=c_delta, edgecolor="white")
        ax5.axhline(0, color="black", linewidth=0.8)
        ax5.set_title("Travel Time vs Most-Likely (min)", fontsize=11)
        ax5.set_ylabel("Δ minutes")
        ax5.spines[["top","right"]].set_visible(False)

    fig.suptitle("CVRPTW KPI Dashboard", fontsize=15, fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    from src.data_loader import load_day
    from src.cvrptw_solver import solve

    data   = load_day(1)
    result = solve(data, scenario="mostlikely", time_limit_s=30)
    if result.is_solved():
        fig = plot_routes(result, data, save_path="outputs/routes_day1.png")
        print("Saved outputs/routes_day1.png")
