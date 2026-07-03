# Multi-UAV Coverage Path Planning under Communication Constraints
**AZAD BHARTI AHIRWAR 240250 **

## Objective
- Engineered an efficiently optimized multi-UAV coverage path planning system under communication constraints.
- Studied and searched for the optimal set of hyperparameter values using the **Optuna** framework across multiple trials.

## Approach
- Implemented a heuristic pipeline combining **BFS**, **DARP + MST**, and **Boustrophedon (Spanning Tree Coverage)** sweeps to manage communication limitations.
- **BFS** repairs disconnected area partitions and checks multi-hop UAV-to-base communication reachability at every timestep.
- **DARP** (Divide Areas based on Robots' Positions) fairly and contiguously splits the map among UAVs using a distance-weighted, iteratively balanced partition.
- **MST (Prim's algorithm)** is built over each UAV's assigned cells; circumnavigating it produces a **Boustrophedon** (lawn-mower) coverage path with zero revisits.
- Validated the algorithm through a Python simulation (grid-based, no physical flight controller required for this stage).

## Result
- Achieved **100% coverage** of free cells with a **0.98/1.00 partition balance score** across UAVs.
- Optuna hyperparameter search (30 trials) tuning communication range and swarm size raised **swarm connectivity from ~6% to ~61%** at equal 100% coverage — showing connectivity, not coverage, is the binding constraint, and that it responds strongly to communication-range tuning.

## Files in this submission
| File | Description |
|---|---|
| `CodeFile.py` | Complete, runnable implementation — GridMap, DARP, BFS, MST/STC, communication model, Optuna tuning, and visualization. Run with `python CodeFile.py`. |
| `Input_Output_and_Jupyter.ipynb` | Executed Jupyter notebook showing the full pipeline running end-to-end, with output graphs and an animated flight-path video (GIF) of the swarm. |
| `README.md` | This file. |

## How to run
```bash
pip install numpy matplotlib optuna
python CodeFile.py
```
This prints mission metrics and saves plots to an `Input & Output/` folder.

To see the graphs, animation, and Optuna study inline, open `Input_Output_and_Jupyter.ipynb` in Jupyter (it is already executed, so outputs are visible without re-running).

## Tech stack
Python, NumPy, Matplotlib, Optuna, Jupyter
