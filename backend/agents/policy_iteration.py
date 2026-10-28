import numpy as np
from utils.models import ValueFunction, Policy
import random
from typing import Any
from exceptions.agent_exceptions import AgentTrainingException

class PolicyIterationAgent:

    def __init__(self, name: str, env_type: str, n_states: int, n_actions: int):
        self._name = name
        self._env_type = env_type
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)
        self._n_states = n_states
        self._n_actions = n_actions

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

    def train(self, train_params: dict[str, Any]) -> ValueFunction:
        required_params = ["P"]

        param_dict = {}

        for param in required_params:
            if param not in train_params:
                raise AgentTrainingException(f"Unable to start the Agent training as the parameter {param} is missing")

            param_dict[param] = train_params[param]

        other_params = ["init_policy", "gamma", "tolerance", "skip_states", "max_iters", "eval_max_iters"]


        for param in other_params:
            if param in train_params:
                param_dict[param] = train_params[param]

        return self._train(**param_dict)


    def _train(self, P: np.ndarray, 
              init_policy: Policy | None = None, 
              gamma: float=0.99, 
              tolerance: float = 1e-9, 
              skip_states: list[int] | None = None, 
              max_iters: int = 1000, 
              eval_max_iters: int = 10_000) -> ValueFunction:
    
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
        pi = init_policy if init_policy is not None else lambda s: random.choice(range(self._n_actions))
        for _ in range(max_iters):
            V = self._evaluate_policy(pi, P, skip_states if skip_states else [], gamma=gamma, tol=tolerance, max_iter=eval_max_iters)
            self._improve_policy(V, P, gamma)
            new_pi = self._select_action
            if all(new_pi(s) == pi(s) for s in range(self._n_states)):
                return V
            pi = new_pi
        raise RuntimeError(f"policy iteration did not converge in {max_iters} rounds")

    def predict(self, state: int):
        return self._select_action(state)