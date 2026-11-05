# Grokking Reinforcement Learning — Implementations

An interactive "Policy Iteration Lab" for a stochastic 1D walk (BanditSlipperyWalk):
a corridor with a goal, a hole, and a probability of slipping to the wrong side on
each move. A FastAPI backend trains a Policy Iteration agent with dynamic
programming, and a vanilla-JS front end lets you configure the environment, train
the agent, play episodes manually or let the trained agent play, and run bulk
predictions.

## Project structure

```
backend/    FastAPI application (entry point: backend/main.py)
            agents/      Policy iteration agent
            registry/    agent / experiment / environment registries
            experiments/ Experiment + model-card logic
            utils/       environments (walk, bandit walk, frozen lake), DP, metrics
frontend/   Static front end (index.html, app.js, style.css) — served by the backend
notebooks/  Jupyter notebooks for the underlying algorithms
```

## Requirements

- Python 3.11+ (developed on 3.13)
- The dependencies live in `requirements.txt` (versions verified against the
  conda environment `rl`).

## Setup

If you already have the `rl` conda environment, just activate it:

```bash
conda activate rl
```

Otherwise create it from scratch:

```bash
conda create -n rl python=3.13 -y
conda activate rl
pip install -r requirements.txt
```

## Starting the application

Run the server from the `backend/` directory (the app imports its own packages
relative to that directory):

```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 8123
```

Add `--reload` for auto-reloading during development. The server serves the API
and the front end on the same port.

## Accessing the application

| What | Where |
| --- | --- |
| Front end (web UI) | http://127.0.0.1:8123/ |
| Interactive API docs (Swagger UI) | http://127.0.0.1:8123/docs |
| ReDoc | http://127.0.0.1:8123/redoc |
| Health check | `GET http://127.0.0.1:8123/health` |

Key API endpoints:

| Endpoint | Description |
| --- | --- |
| `POST /register_experiment` | Register a new experiment (agent + environment) |
| `GET /experiments` | List all registered experiments |
| `GET /experiments/filter?agent_type=&env_type=&trained=&env_id=` | Filter experiments |
| `GET /experiments/{experiment_id}` | Experiment details + learned value function |
| `GET /envs`, `GET /envs/{env_id}` | List / fetch environment configs |
| `POST /train` | Train the experiment's agent, returns V(s) |
| `POST /predict` | Greedy action for a single state |
| `WS /ws/predict` | Persistent WebSocket channel for bulk predictions |

Note: registries are in-memory — experiments are lost when the server restarts.

## Using the front end

The UI has three numbered steps (left column), the corridor and value plot
(center column), and play controls (right column).

### 1. Environment setup

Configure the corridor, then click **Initialize experiment**:

- **Walk length** — number of states (3–20)
- **Start / Hole / Goal position** — indices along the corridor
- **Slip probability** — chance a move goes the wrong way
- **Seed** — RNG seed for the environment
- **Goal / Hole reward** — reward values for terminal states

This registers a new experiment in the backend registry and renders the
corridor (G = goal, H = hole, A = agent position).

### 2. Training

Set the training hyper-parameters and click **Train & plot V(s)**:

- **γ (gamma)** — discount factor
- **Tolerance** — value-iteration convergence threshold
- **Initial policy** — `random`, `always_left`, or `always_right`

After training you get:

- the learned **value function V(s)** plotted in the center column,
- policy arrows (← / →) drawn on the corridor,
- a **model card** summarizing the experiment (params, env_id, trained flag).

### 3. Play

Pick a mode and take actions; **Reset episode** restarts the walk.

- **Manual** — move with the LEFT / RIGHT buttons or the ← / → keyboard keys.
  Live stats show the current state, step count, cumulative reward, and the
  episode result (goal / hole).
- **AI predict** — the trained agent plays on its own, taking a greedy action
  every step. Adjust the **AI step interval** slider to change playback speed.
  AI mode requires a trained model for the *current* environment.

### Session library

Lists every experiment registered this server session. Pick one and click
**Load model** to restore its environment and learned policy — required before
using **AI predict** with a model you loaded rather than just trained.

### Bulk prediction

Runs a configurable number of full episodes (1–10,000) of the trained policy
over a persistent WebSocket connection, then reports:

- successes / total episodes
- win rate
- average steps per episode
