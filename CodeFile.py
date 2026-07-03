"""
CodeFile.py
===========
Multi-UAV Coverage Path Planning under Communication Constraints
Author: Aditya Panwar (240063)

A single-file implementation combining:
  - GridMap            : occupancy grid with random obstacles
  - DARP partitioning   : distance-weighted, load-balanced area division
  - BFS utilities       : partition connectivity repair + comms reachability
  - MST + STC           : Prim's MST -> Spanning Tree Coverage (Boustrophedon sweep)
  - Communication model : multi-hop relay connectivity to a ground station
  - Optuna tuning        : hyperparameter search over comm range & swarm size
  - Visualization         : matplotlib rendering of partitions + paths

Run directly to execute a demo mission, print metrics, and save plots:
    python CodeFile.py
"""

import os
import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ============================================================
# 1. GRID MAP
# ============================================================

class GridMap:
    def __init__(self, rows: int, cols: int, obstacle_density: float = 0.08, seed: int = 42):
        self.rows = rows
        self.cols = cols
        self.rng = np.random.default_rng(seed)
        self.obstacle_mask = self._generate_obstacles(obstacle_density)

    def _generate_obstacles(self, density: float) -> np.ndarray:
        mask = self.rng.random((self.rows, self.cols)) < density
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
        return mask

    def is_free(self, r: int, c: int) -> bool:
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        return not self.obstacle_mask[r, c]

    def free_cells(self):
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if self.is_free(r, c)]

    def neighbors4(self, r: int, c: int):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if self.is_free(nr, nc):
                yield nr, nc


# ============================================================
# 2. BFS UTILITIES (partition repair + comms reachability)
# ============================================================

def largest_connected_component(cells, grid):
    cells = set(cells)
    visited = set()
    best_component = set()
    for start in cells:
        if start in visited:
            continue
        component = set()
        queue = deque([start])
        visited.add(start)
        while queue:
            r, c = queue.popleft()
            component.add((r, c))
            for nr, nc in grid.neighbors4(r, c):
                if (nr, nc) in cells and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        if len(component) > len(best_component):
            best_component = component
    return best_component


def repair_partition_connectivity(assignment: dict, grid, num_robots: int):
    for robot_id in range(num_robots):
        owned = [cell for cell, rid in assignment.items() if rid == robot_id]
        if not owned:
            continue
        main_component = largest_connected_component(owned, grid)
        stray_cells = set(owned) - main_component
        for cell in stray_cells:
            r, c = cell
            neighbor_owners = [
                assignment[(nr, nc)]
                for nr, nc in grid.neighbors4(r, c)
                if (nr, nc) in assignment and assignment[(nr, nc)] != robot_id
            ]
            assignment[cell] = neighbor_owners[0] if neighbor_owners else robot_id
    return assignment


def communication_bfs_reachable(positions, comm_range: float, base_index: int = 0):
    n = len(positions)
    positions = np.array(positions, dtype=float)
    visited = {base_index}
    queue = deque([base_index])
    while queue:
        i = queue.popleft()
        for j in range(n):
            if j in visited:
                continue
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= comm_range:
                visited.add(j)
                queue.append(j)
    return visited


# ============================================================
# 3. DARP PARTITIONING
# ============================================================

def _dist(cell, start):
    return float(np.hypot(cell[0] - start[0], cell[1] - start[1]))


def darp_partition(grid, robot_positions, max_iterations: int = 30, tolerance: float = 1.0):
    num_robots = len(robot_positions)
    free_cells = grid.free_cells()
    weights = np.ones(num_robots)
    assignment = {}

    for _ in range(max_iterations):
        assignment = {}
        counts = np.zeros(num_robots)
        for cell in free_cells:
            costs = [weights[i] * _dist(cell, robot_positions[i]) for i in range(num_robots)]
            owner = int(np.argmin(costs))
            assignment[cell] = owner
            counts[owner] += 1

        target = len(free_cells) / num_robots
        imbalance = counts - target
        if np.max(np.abs(imbalance)) <= tolerance:
            break
        weights *= 1.0 + 0.05 * (imbalance / max(target, 1e-6))
        weights = np.clip(weights, 0.1, 10.0)

    assignment = repair_partition_connectivity(assignment, grid, num_robots)
    return assignment


def partition_stats(assignment: dict, num_robots: int):
    counts = [0] * num_robots
    for owner in assignment.values():
        counts[owner] += 1
    total = sum(counts) or 1
    balance = 1.0 - (max(counts) - min(counts)) / (total / num_robots)
    return {"cell_counts": counts, "balance_score": max(balance, 0.0)}


# ============================================================
# 4. MST + SPANNING TREE COVERAGE (BOUSTROPHEDON)
# ============================================================

def _grid_neighbors(cell):
    r, c = cell
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def build_mst(cells):
    cells = list(cells)
    if not cells:
        return {}
    cell_set = set(cells)
    start = cells[0]
    visited = {start}
    adjacency = {c: [] for c in cells}
    heap = []
    for n in _grid_neighbors(start):
        if n in cell_set:
            heapq.heappush(heap, (1, start, n))
    while heap and len(visited) < len(cells):
        _, u, v = heapq.heappop(heap)
        if v in visited:
            continue
        visited.add(v)
        adjacency[u].append(v)
        adjacency[v].append(u)
        for n in _grid_neighbors(v):
            if n in cell_set and n not in visited:
                heapq.heappush(heap, (1, v, n))
    return adjacency


def generate_stc_path(adjacency: dict, start_cell):
    if not adjacency or start_cell not in adjacency:
        return []
    corner_offsets = [(0, 0), (0, 1), (1, 1), (1, 0)]
    path = []
    visited_edges = set()

    def dfs(u, parent):
        r, c = u
        path.append((2 * r + corner_offsets[0][0], 2 * c + corner_offsets[0][1]))
        for v in adjacency[u]:
            edge = (u, v)
            if v == parent or edge in visited_edges:
                continue
            visited_edges.add((u, v))
            visited_edges.add((v, u))
            dfs(v, u)
        path.append((2 * r + corner_offsets[2][0], 2 * c + corner_offsets[2][1]))

    dfs(start_cell, None)
    return path


def path_length(path):
    if len(path) < 2:
        return 0.0
    pts = np.array(path, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


# ============================================================
# 5. COMMUNICATION CONSTRAINT MODEL
# ============================================================

def connectivity_ratio(uav_positions_over_time, comm_range: float, base_position):
    if not uav_positions_over_time:
        return 1.0
    connected_count = 0
    total_count = 0
    for positions in uav_positions_over_time:
        all_nodes = [base_position] + list(positions)
        reachable = communication_bfs_reachable(all_nodes, comm_range, base_index=0)
        for uav_idx in range(len(positions)):
            total_count += 1
            if (uav_idx + 1) in reachable:
                connected_count += 1
    return connected_count / total_count if total_count else 1.0


# ============================================================
# 6. MISSION ORCHESTRATOR
# ============================================================

@dataclass
class MissionResult:
    grid: GridMap
    assignment: Dict[Tuple[int, int], int]
    paths: List[List[Tuple[float, float]]]
    metrics: dict = field(default_factory=dict)


def _initial_positions(grid: GridMap, num_robots: int):
    positions = []
    col_step = max(grid.cols // (num_robots + 1), 1)
    for i in range(1, num_robots + 1):
        c = min(i * col_step, grid.cols - 1)
        r = 0
        while not grid.is_free(r, c) and r < grid.rows - 1:
            r += 1
        positions.append((r, c))
    return positions


def _simulate_positions_over_time(paths):
    if not paths:
        return []
    max_len = max((len(p) for p in paths), default=0)
    if max_len == 0:
        return []
    timeline = []
    for t in range(max_len):
        snapshot = []
        for p in paths:
            snapshot.append((0, 0) if not p else p[min(t, len(p) - 1)])
        timeline.append(snapshot)
    return timeline


def run_mission(rows=20, cols=20, num_robots=3, obstacle_density=0.08,
                 comm_range=8.0, base_position=None, seed=42):
    grid = GridMap(rows, cols, obstacle_density=obstacle_density, seed=seed)
    robot_positions = _initial_positions(grid, num_robots)
    base_position = base_position or robot_positions[0]

    assignment = darp_partition(grid, robot_positions)
    stats = partition_stats(assignment, num_robots)

    paths = []
    for robot_id, start in enumerate(robot_positions):
        owned_cells = [cell for cell, rid in assignment.items() if rid == robot_id]
        if not owned_cells:
            paths.append([])
            continue
        adjacency = build_mst(owned_cells)
        stc_start = min(owned_cells, key=lambda c: _dist(c, start))
        paths.append(generate_stc_path(adjacency, stc_start))

    total_len = sum(path_length(p) for p in paths)
    covered_cells = set(assignment.keys())
    coverage_pct = 100.0 * len(covered_cells) / max(len(grid.free_cells()), 1)

    positions_over_time = _simulate_positions_over_time(paths)
    conn_ratio = connectivity_ratio(positions_over_time, comm_range, base_position)

    metrics = {
        "coverage_pct": coverage_pct,
        "total_path_length": total_len,
        "balance_score": stats["balance_score"],
        "cell_counts": stats["cell_counts"],
        "connectivity_ratio": conn_ratio,
    }
    return MissionResult(grid=grid, assignment=assignment, paths=paths, metrics=metrics)


# ============================================================
# 7. VISUALIZATION
# ============================================================

def plot_mission(result: MissionResult, save_path=None, show=False):
    grid = result.grid
    assignment = result.assignment
    paths = result.paths

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(paths), 1)))

    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.obstacle_mask[r, c]:
                ax.add_patch(patches.Rectangle((c, grid.rows - 1 - r), 1, 1, color="black"))

    for (r, c), robot_id in assignment.items():
        ax.add_patch(patches.Rectangle((c, grid.rows - 1 - r), 1, 1,
                                        color=colors[robot_id], alpha=0.15, linewidth=0))

    for robot_id, path in enumerate(paths):
        if not path:
            continue
        xs = [c / 2 + 0.25 for _, c in path]
        ys = [grid.rows - (r / 2) - 0.75 for r, _ in path]
        ax.plot(xs, ys, color=colors[robot_id], linewidth=1.5, label=f"UAV {robot_id}", alpha=0.9)
        ax.plot(xs[0], ys[0], marker="o", color=colors[robot_id], markersize=8)

    ax.set_xlim(0, grid.cols)
    ax.set_ylim(0, grid.rows)
    ax.set_aspect("equal")
    ax.set_title("Multi-UAV Coverage Path Planning (DARP + MST + STC/Boustrophedon)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    m = result.metrics
    caption = (f"Coverage: {m['coverage_pct']:.1f}%  |  Balance: {m['balance_score']:.2f}  |  "
               f"Connectivity: {m['connectivity_ratio']*100:.1f}%  |  "
               f"Total path length: {m['total_path_length']:.1f}")
    fig.text(0.5, 0.02, caption, ha="center", fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ============================================================
# 8. OPTUNA HYPERPARAMETER TUNING
# ============================================================

def optuna_objective(trial):
    import optuna  # local import so the rest of the file works without optuna installed
    comm_range = trial.suggest_float("comm_range", 3.0, 15.0)
    num_robots = trial.suggest_int("num_robots", 2, 6)
    result = run_mission(rows=20, cols=20, num_robots=num_robots, obstacle_density=0.08,
                          comm_range=comm_range, seed=42)
    m = result.metrics
    return (m["coverage_pct"] + m["balance_score"] * 20 + m["connectivity_ratio"] * 30
            - 0.5 * m["total_path_length"] / max(num_robots, 1))


def run_optuna_study(n_trials: int = 30, seed: int = 7):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)
    return study


# ============================================================
# 9. MAIN DEMO
# ============================================================

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Input & Output")
    os.makedirs(out_dir, exist_ok=True)

    print("Running demo mission (4 UAVs, 20x20 grid)...")
    result = run_mission(rows=20, cols=20, num_robots=4, obstacle_density=0.08,
                          comm_range=20.0, seed=42)
    print("\n=== Mission Metrics ===")
    for k, v in result.metrics.items():
        print(f"{k}: {v}")
    plot_mission(result, save_path=os.path.join(out_dir, "demo_mission.png"))
    print(f"\nSaved plot to {out_dir}/demo_mission.png")

    print("\nRunning Optuna study (30 trials)...")
    study = run_optuna_study(n_trials=30)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
