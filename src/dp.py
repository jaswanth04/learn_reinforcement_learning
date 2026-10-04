"""Dynamic programming: policy evaluation, improvement, and iteration."""

import random

import numpy as np

from models import Environment, Policy, QTable, ValueFunction


# ---------------------------------------------------------------- helpers
def default_policy(env: Environment, seed: int | None = None) -> Policy:
    """Uniform random policy — a neutral starting point for policy iteration."""
    rng = random.Random(seed if seed is not None else 0)
    codes = list(env.actions)
    return lambda s: int(rng.choice(codes))


# ---------------------------------------------------------------- evaluation
def policy_evaluation(env: Environment, pi: Policy, gamma: float = 0.99,
                      tol: float = 1e-9, max_iter: int = 10_000) -> ValueFunction:
    """Iterative policy evaluation (DP).

    V(s) = Σ p(s',r|s,a)[ r + γ(1-done) V(s') ]   for a = pi(s)

    `max_iter` bounds the number of full sweeps; the loop also stops
    early once the largest per-state change is below `tol`.
    """
    V = np.zeros(env.n_states, dtype=np.float64)
    for _ in range(max_iter):
        delta = 0.0
        for s in range(env.n_states):
            if s in env.holes or s == env.goal:
                continue                      # absorbing: V stays exactly 0
            v = sum(
                o.prob * (o.reward + (0.0 if o.done else gamma * V[o.next_state]))
                for o in env.P[s][pi(s)]
            )
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    return V


def solve_policy_value(env: Environment, pi: Policy, gamma: float = 0.99,
                       tol: float | None = None, max_iter: int | None = None) -> ValueFunction:
    """EXACT policy evaluation via linear algebra.

    (I - γ P_π) V = R_π   over the non-terminal states (terminals have V=0).

    One shot, no iterative sweeps. `tol`/`max_iter` are accepted so this
    can be swapped in wherever policy_evaluation is used (they are ignored).
    """
    terms = set(env.holes) | {env.goal}
    idx = [s for s in range(env.n_states) if s not in terms]
    k = len(idx)

    R = np.zeros(k, dtype=np.float64)
    P = np.zeros((k, k), dtype=np.float64)

    for i, s in enumerate(idx):
        R[i] = sum(o.prob * o.reward for o in env.P[s][pi(s)])
        for j, t in enumerate(idx):
            P[i, j] = sum(
                o.prob for o in env.P[s][pi(s)]
                if o.next_state == t and not o.done
            )

    V_int = np.linalg.solve(np.eye(k) - gamma * P, R)

    V = np.zeros(env.n_states, dtype=np.float64)
    for i, s in enumerate(idx):
        V[s] = V_int[i]
    return V


# ---------------------------------------------------------------- improvement
def policy_improvement(env: Environment, V: ValueFunction,
                       gamma: float = 0.99) -> Policy:
    """One greedy-improvement step (policy-improvement theorem).

    Q(s,a) = Σ p(s'|s,a)[ r + γ(1-done) V(s') ]
    π'(s)  = argmax_a Q(s,a)
    """
    Q: QTable = np.zeros((env.n_states, len(env.actions)), dtype=np.float64)
    for s in range(env.n_states):
        for a in env.P[s]:
            Q[s, a] = sum(
                o.prob * (o.reward + (0.0 if o.done else gamma * V[o.next_state]))
                for o in env.P[s][a]
            )

    greedy = np.argmax(Q, axis=1)
    return lambda s: int(greedy[s])


# ---------------------------------------------------------------- iteration
def policy_iteration(env: Environment, gamma: float = 0.99, tol: float = 1e-9,
                     max_iters: int = 1000, eval_max_iters: int = 10_000,
                     init_policy: Policy | None = None,
                     evaluator: callable = policy_evaluation) -> tuple[Policy, ValueFunction]:
    """Full policy iteration: evaluate -> improve -> ... until stable.

    Stability of the greedy policy = Bellman optimality for a finite MDP,
    so the returned (pi, V) is optimal.

    - `init_policy`   : starting policy (default: seeded uniform random).
                        Any start converges; this only affects iteration count.
    - `max_iters`     : cap on OUTER evaluate/improve rounds.
    - `eval_max_iters`: cap on the INNER sweeps of each evaluation
                        (modified policy iteration if small; exact if large).
    - `evaluator`     : plug in solve_policy_value for exact inner solves.
    """
    pi = init_policy if init_policy is not None else default_policy(env, seed=0)
    for _ in range(max_iters):
        V = evaluator(env, pi, gamma=gamma, tol=tol, max_iter=eval_max_iters)
        new_pi = policy_improvement(env, V, gamma=gamma)
        if all(new_pi(s) == pi(s) for s in range(env.n_states)):
            return pi, V
        pi = new_pi
    raise RuntimeError(f"policy iteration did not converge in {max_iters} rounds")