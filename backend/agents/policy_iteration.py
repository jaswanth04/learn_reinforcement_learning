import numpy as np
from utils.models import ValueFunction, Policy
import random
from typing import Any, Callable
from exceptions.agent_exceptions import AgentInitializationException

from dataclasses import dataclass, fields

from typing import Self


@dataclass
class PolicyIterationTrainingParameters:
    P: np.ndarray
    init_policy: str | None = None
    gamma: float = 0.99
    tolerance: float = 1e-9
    skip_states: list[int] | None = None
    max_iters: int = 1000
    eval_max_iters: int = 10_000

    @classmethod
    def from_dict(cls, train_params: dict[str, Any]) -> Self:
        """Create the dataclass from a dict, skipping any missing/invalid keys.
        Missing keys fall back to the dataclass defaults."""

        if "P" not in train_params:
            raise AgentInitializationException("P is not provided for an Agent based on MDP")

        known_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in train_params.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class PolicyIterationAgentParameters:
    name: str
    env_type: str
    n_states: int
    n_actions: int

    @classmethod
    def from_dict(cls, agent_params: dict[str, Any]) -> Self:
        known_fields = {f.name for f in fields(cls)}

        missing = [f.name for f in fields(cls) if f.name not in agent_params]
        if missing:
            raise AgentInitializationException(
                f"Missing required agent parameters: {', '.join(missing)}"
            )

        filtered = {k: v for k, v in agent_params.items() if k in known_fields}
        return cls(**filtered)


class PolicyIterationAgent:

    def __init__(self, params: PolicyIterationAgentParameters):
        self._name = params.name
        self._env_type = params.env_type
        self.Q = np.zeros((params.n_states, params.n_actions), dtype=np.float64)
        self._n_states = params.n_states
        self._n_actions = params.n_actions

        self._init_policy_map: dict[str, Callable[[Any], int]] = {
                "random": lambda _: random.choice(range(self._n_actions)),
                "always_left": lambda _: 0,
                "always_right": lambda _: 1
            }

    @property
    def name(self):
        return self._name

    def _evaluate_policy(self, pi: Policy,
                         P: np.ndarray,
                         skip_states: list[int],
                         gamma: float = 0.99,
                         tol: float = 1e-9,
                         max_iter: int = 10_000) -> ValueFunction:

        """Iterative policy evaluation (DP).

            V(s) = Σ p(s',r|s,a)[ r + γ(1-done) V(s') ]   for a = pi(s)

            `max_iter` bounds the number of full sweeps; the loop also stops
            early once the largest per-state change is below `tol`.
        """
        V = np.zeros(self._n_states, dtype=np.float64)
        for _ in range(max_iter):
            delta = 0.0
            for s in range(self._n_states):
                if s in skip_states:
                    continue                      # absorbing: V stays exactly 0
                v = sum(
                    o.prob * (o.reward + (0.0 if o.done else gamma * V[o.next_state]))
                    for o in P[s][pi(s)]
                )
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            if delta < tol:
                break
        return V

    def _improve_policy(self, V: ValueFunction,
                       P: np.ndarray,
                       gamma: float = 0.99):
        """One greedy-improvement step (policy-improvement theorem).

        Q(s,a) = Σ p(s'|s,a)[ r + γ(1-done) V(s') ]
        π'(s)  = argmax_a Q(s,a)
        """
        for s in range(self._n_states):
            for a in P[s]:
                self.Q[s, a] = sum(
                    o.prob * (o.reward + (0.0 if o.done else gamma * V[o.next_state]))
                    for o in P[s][a]
                )

    def _select_action(self, state: int) -> int:
        greedy = np.argmax(self.Q, axis=1)
        return int(greedy[state])

    def train(self, train_params: PolicyIterationTrainingParameters) -> ValueFunction:
        """Full policy iteration: evaluate -> improve -> ... until stable.

        Stability of the greedy policy = Bellman optimality for a finite MDP,
        so the returned (pi, V) is optimal.

        - `init_policy`    : starting policy (default: seeded uniform random).
                             Any start converges; this only affects iteration count.
        - `max_iters`      : cap on OUTER evaluate/improve rounds.
        - `eval_max_iters` : cap on the INNER sweeps of each evaluation
                             (modified policy iteration if small; exact if large).
        - `evaluator`      : plug in solve_policy_value for exact inner solves.
        """
        skip_states = train_params.skip_states if train_params.skip_states else []
        pi = self._create_policy(train_params.init_policy)

        for _ in range(train_params.max_iters):
            V = self._evaluate_policy(
                pi,
                train_params.P,
                skip_states,
                gamma=train_params.gamma,
                tol=train_params.tolerance,
                max_iter=train_params.eval_max_iters,
            )
            self._improve_policy(V, train_params.P, train_params.gamma)
            new_pi = self._select_action
            if all(new_pi(s) == pi(s) for s in range(self._n_states)):
                return V
            pi = new_pi

        raise RuntimeError(f"policy iteration did not converge in {train_params.max_iters} rounds")

    def _create_policy(self, policy_type: str | None) -> Callable[[Any], int]:

        if policy_type is None or policy_type not in self._init_policy_map:
            print("Selecting default policy")
            return self._init_policy_map["random"]

        return self._init_policy_map[policy_type]

    def predict(self, state: int):
        return self._select_action(state)