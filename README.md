# Grokking Reinforcement Learning — Implementations

An interactive policy-iteration lab for gridworld-style MDPs. It ships three
environments — a stochastic 1D **Bandit Slippery Walk** (a corridor with a goal,
a hole, and a slip probability), the classic slippery **Frozen Lake** grid (4×4
plus a gym-style 8×8 map), and the textbook **Russell & Norvig** 3×4 gridworld
(a +1 goal, a −1 pit, an impassable wall, and −0.04 per step) — each with its
own web page. Pick a training agent per experiment: a **Policy Iteration** agent
trained with dynamic programming over the MDP; a model-free TD-control agent —
**SARSA** (on-policy), **Q-Learning** or **Double Q-Learning** (off-policy) —
that needs no model and learns live, one step at a time, over a WebSocket; or a
Dyna-style **online planning** agent — **Dyna-Q** or **Trajectory Sampling** —
that learns a model of the environment from its own experience and plans on top
of it. A vanilla-JS front end lets you configure the environment, train the
agent, play episodes manually or let the trained agent play, and run bulk
predictions.

## Project structure

```
backend/    FastAPI application (entry point: backend/main.py)
            agents/      Policy Iteration + TD control (SARSA, Q-Learning,
                         Double Q-Learning) + online planning (Dyna-Q,
                         Trajectory Sampling)
            core/        shared types (ValueFunction, Environment protocol, ...)
            registry/    agent / experiment / environment registries
            experiments/ Experiment + model-card logic
            utils/       environments (walk, bandit walk, frozen lake,
                         russell-norvig gridworld), DP, policy_control
                         (SARSA, exploration), policy_estimation
frontend/   Static front end — served by the backend
            index.html          environment chooser (home page)
            bandit_walk.html    Bandit Slippery Walk lab
            frozen_lake.html    Frozen Lake lab
            rn_gridworld.html   Russell & Norvig Grid World lab
            bandit_walk.js      bandit walk page logic
            frozen_lake.js      frozen lake page logic
            rn_gridworld.js     russell & norvig grid world page logic
            style.css           shared styles
notebooks/  Jupyter notebooks for the underlying algorithms
```

## Requirements

- Python 3.11+ (developed on 3.13)
- The dependencies live in `requirements.txt` (versions verified against the
  conda environment `rl`).

## Setup

Pick either option A (conda) or option B (plain Python venv).

### Option A: conda

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

### Option B: Python virtual environment

From the project root, create and activate a venv, then install the
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The venv is created at `.venv/` in the project root (already in `.gitignore`).
Deactivate it any time with `deactivate`.

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
| Environment chooser (home) | http://127.0.0.1:8123/ |
| Bandit Slippery Walk lab | http://127.0.0.1:8123/bandit_walk.html |
| Frozen Lake lab | http://127.0.0.1:8123/frozen_lake.html |
| Russell & Norvig Grid World lab | http://127.0.0.1:8123/rn_gridworld.html |
| Interactive API docs (Swagger UI) | http://127.0.0.1:8123/docs |
| ReDoc | http://127.0.0.1:8123/redoc |
| Health check | `GET http://127.0.0.1:8123/health` |

The home page (`/`) lists the available environments. Pick one to open its lab;
each lab page has a **← All environments** link in the header to go back.

Key API endpoints:

| Endpoint | Description |
| --- | --- |
| `GET /health` | Server liveness |
| `GET /agents` | List registered agent types (`agent_type`, `description`, `stepwise`) — the front-end dropdown is built from this |
| `POST /register_experiment` | Register a new experiment (`agent_type`: `PolicyIteration`, `Sarsa`, `QLearning`, `DoubleQLearning`, `DynaQ`, or `TrajectorySampling`) |
| `GET /experiments` | List all registered experiments |
| `GET /experiments/filter?agent_type=&env_type=&trained=&env_id=` | Filter experiments |
| `GET /experiments/{experiment_id}` | Experiment details + learned value function + episodes trained |
| `GET /envs`, `GET /envs/{env_id}` | List / fetch environment configs |
| `POST /train` | Train a Policy Iteration agent, returns V(s) |
| `WS /ws/train_step` | Step-wise training channel for the interaction-based agents — TD (SARSA / Q-Learning / Double Q-Learning) and online planning (Dyna-Q / Trajectory Sampling): `start` / `step` / `episode` / `finish` |
| `POST /predict` | Greedy action for a single state |
| `WS /ws/predict` | Persistent WebSocket channel for bulk predictions |

Note: registries are in-memory — experiments are lost when the server restarts.

Note: **Dyna-Q** and **Trajectory Sampling** are online-planning agents — they
learn a model of the environment as they go and do extra planning updates per
real step, so a training run takes noticeably longer than the other learning
agents (more so on the 8×8 lake). The Policy Iteration agent still solves the
MDP exactly in one click regardless of board size.

## Using the front end

Each lab shares the same layout and workflow: setup + training (left column),
board + value plot (center column), and play + session tools (right column).
The steps below apply to all three environments; the environment-specific setup
options are noted under step 1.

### 1. Environment setup

Configure the environment, then click **Initialize experiment**. This registers
a new experiment in the backend registry and renders the board.

**Bandit Slippery Walk** (`/bandit_walk.html`):

- **Walk length** — number of states (3–20)
- **Start / Hole / Goal position** — indices along the corridor
- **Slip probability** — chance a move goes the wrong way
- **Seed** — RNG seed for the environment
- **Goal / Hole reward** — reward values for terminal states

The corridor renders with `·` = walkable, `G` = goal, `H` = hole, `A` = agent
position.

**Frozen Lake** (`/frozen_lake.html`):

- **Board layout** — preset grids: Classic 4×4, Classic 8×8 (10 holes),
  Easy 4×4, Tricky 4×4, Wide 5×5, Narrow 3×3, or Random 4×4 (holes derived
  from the seed)
- **Intended-move probability** — chance the move goes where you aimed.
  `1.000` = deterministic, `0.333` = classic ice (intended / left / right each
  ⅓), lower = more slippery. The remaining probability is split evenly between
  the two perpendicular slip directions.
- **Goal / Hole reward** — reward values for terminal states
- **Seed** — RNG seed (also drives the Random layout)

The grid renders with `S` = start, `G` = goal, `H` = hole, `F` = frozen,
`A` = agent position.

**Russell & Norvig Grid World** (`/rn_gridworld.html`):

- **Board layout** — the standard AIMA 3×4 grid: start `S` bottom-left, an
  impassable wall `#` in the middle, a `+1` goal top-right and a `−1` pit below
  it.
- **Intended-move probability** — chance the move goes where you aimed.
  `0.800` = classic AIMA (intended 80%, each perpendicular side 10%), `1.000` =
  deterministic, lower = more erratic.
- **Goal reward (+) / Pit reward (−) / Step cost** — terminal rewards and the
  per-step living reward (default −0.04).
- **Seed** — RNG seed.

Unlike Frozen Lake, the wall is **impassable** (a move into it, or off the grid,
leaves the agent in place) and there are **two** terminals — a `+1` goal and a
`−1` pit — both absorbing, with every non-terminal step costing −0.04.

The grid renders with `S` = start, `.` = open, `#` = wall, `+` = goal (+1),
`-` = pit (−1), `A` = agent position.

### 2. Training

Pick the **Training agent** first — the hyper-parameter panel and the train
button switch accordingly, and the experiment is registered with the selected
agent (the backend builds the matching agent instance):

- **Policy Iteration (needs the MDP)** — model-based dynamic programming;
  solves for the optimal policy in one click.
- **SARSA (learns by stepping)** — model-free on-policy TD control; learns
  live from interaction over a WebSocket.
- **Q-Learning (learns by stepping)** — model-free off-policy TD control;
  bootstraps from max_a Q(s', a), so it learns the optimal policy directly.
- **Double Q-Learning (learns by stepping)** — off-policy TD control with two
  tables (Q1/Q2); selects the next action with one table and evaluates it with
  the other to reduce the maximisation bias. Q = (Q1 + Q2) / 2.
- **Dyna-Q (learns by stepping + planning)** — off-policy Q-Learning that also
  builds a model of the environment from every real transition and, after each
  step, runs `n_planning` extra Q-Learning updates on (state, action) pairs
  sampled from that model.
- **Trajectory Sampling (learns by stepping + planning)** — Dyna-style planning
  that, instead of independently sampled transitions, replays a single simulated
  *trajectory* through the learned model (up to `max_trajectory_depth` steps)
  after every real step, restricting each rollout to actions the model has
  actually seen in that state.

**Policy Iteration** — click **Train & plot V(s)**:

- **γ (gamma)** — discount factor
- **Tolerance** — value-iteration convergence threshold
- **Initial policy** — `random` or a fixed action. Policy iteration converges
  to the optimal policy from any start; the initial policy only affects the
  iteration count.

After training you get:

- the learned **value function V(s)** plotted in the center column (goal and
  hole bars are highlighted),
- policy arrows drawn on the board (← / → for the walk; ← ↑ ↓ → for the grid),
- a **model card** summarizing the experiment (params, env_id, trained flag).

**SARSA / Q-Learning / Double Q-Learning / Dyna-Q / Trajectory Sampling** —
click **Train `<agent>` (live)** (the button is labelled with the selected
agent; click it again to **Stop training**):

- **γ (gamma)** — discount factor (default 1.0 for the pure TD agents; the two
  planning agents drop it automatically — to 0.95, or 0.99 on the 8×8 lake —
  because replaying zero-reward wall-bump self-loops from the learned model
  inflates Q at γ = 1)
- **Episodes** — how many episodes to train this run (front-end controlled)
- **Max steps / episode** — an episode is force-reset when this budget runs out
- **init α / min α / α decay ratio** — learning-rate schedule (indexed by episode)
- **init ε / min ε / ε decay ratio** — ε-greedy exploration schedule (by episode)
- **Planning steps (n)** — Dyna-Q only: extra model-based updates run per real
  step (default 30)
- **Max trajectory depth** — Trajectory Sampling only: how long the simulated
  rollout replayed through the model after each real step may be (default 100)
- **Initial policy** — behaviour policy used before any update

> **Dyna-Q and Trajectory Sampling learn the environment, so they are slower.**
> On top of the Q-Learning update they keep a count/reward model of the
> environment and run extra planning updates on simulated experience after every
> real step — more compute per step than plain SARSA or Q-Learning, so a run
> takes longer, and on a big sparse board like Frozen Lake 8×8 it also needs
> many more episodes (raise Episodes to 10 000+ and train with γ = 0.99).

The front end owns the rollouts and drives `WS /ws/train_step`: each observed
`(state, action, reward, next_state, done)` is sent to the server, which
applies the agent's update (the TD update for SARSA / Q-Learning / Double
Q-Learning; a model update plus planning updates for Dyna-Q / Trajectory
Sampling) and replies with the next action, the live value function
V(s) = max_a Q(s,a), and the current episode / step / α / ε. While
training you get:

- a **live line chart** — one line per state, x-axis = episode, updated every
  episode (goal and hole lines are highlighted),
- V(s) figures on every board cell and a live **Episode / Step / α / ε** strip,
- learned policy arrows + model card once the run finishes.

Trained semantics and warm starts:

- Only a **fully completed** run (or a deliberate Stop at an episode
  boundary followed by completion) marks the model **trained** — an
  interrupted run keeps its learning but is listed as *not trained*.
- Training again on the same experiment is a **warm start**: the Q-table is
  kept and training continues on top of it (α/ε schedules restart per run).
  Click **Initialize experiment** for a clean slate.
- The model card reports **Episodes trained** — the cumulative count across
  all warm-start runs on that Q-table.
- While any training runs, **Load model** and **Run bulk prediction** are
  disabled until the run completes or is stopped.

### 3. Play

Pick a mode and take actions; **Reset episode** restarts the episode.

- **Manual** — act with the on-screen buttons or the keyboard. Bandit walk:
  LEFT / RIGHT buttons or the ← / → keys. Frozen Lake and Russell & Norvig: the
  4-way pad or the arrow keys / WASD. Live stats show the current state, step
  count, cumulative reward, and episode result (goal / hole). A "last move" line
  reports whether the move was as intended, a slip, or a wall/edge bounce.
- **AI predict** — the trained agent plays on its own, taking a greedy action
  every step. Adjust the **AI step interval** slider to change playback speed.
  AI mode requires a trained model for the *current* environment.

### Session library

Lists every experiment registered this server session (all three environments).
Pick one and click **Load model** to restore that model's environment and
learned policy — required before using **AI predict** with a model you loaded
rather than just trained.

### Bulk prediction

Runs a configurable number of full episodes (1–10,000) of the trained policy
over a persistent WebSocket connection, then reports:

- successes / total episodes
- win rate
- average steps per episode
