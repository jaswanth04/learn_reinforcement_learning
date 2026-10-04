
"""Shared reset/step/seed runtime for all environments."""

import random

from models import Action, ActionResult


class EnvCore:
    """Base class for every environment's *interaction* logic.

    Subclasses must provide before calling reset():
      - self._actions:          dict[int, Action]
      - self._actions_by_name:  dict[str, int]
      - self._P:                ProbabilityTransition
      - self.start_state:       int
      - self._state:            int
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)          # env owns its RNG

    # ---- rng ---------------------------------------------------------------
    def seed(self, seed: int | None = None) -> None:
        """Re-seed the env's own RNG for reproducible experiments."""
        self._rng.seed(seed)

    # ---- interaction --------------------------------------------------------
    def reset(self) -> int:
        self._state = self.start_state
        return self._state

    def step(self, action: int | Action | str) -> ActionResult:
        """Take an action; return the realized outcome as ActionResult(prob=None).

        The probability is None because sampling has already happened;
        the probabilities live in P[s][a].
        """
        code = self._resolve_action(action)
        if code not in self._P[self._state]:
            raise ValueError(f"invalid action {action!r} from state {self._state}")

        outcome = self._sample_outcome(self._P[self._state][code])
        self._state = outcome.next_state
        return ActionResult(prob=None,
                            next_state=outcome.next_state,
                            reward=outcome.reward,
                            done=outcome.done)

    def _resolve_action(self, action: int | Action | str) -> int:
        if isinstance(action, Action):
            return action.code
        if isinstance(action, str):
            if action not in self._actions_by_name:
                raise ValueError(f"unknown action name {action!r}")
            return self._actions_by_name[action]
        return int(action)

    def _sample_outcome(self, outcomes: list[ActionResult]) -> ActionResult:
        if len(outcomes) == 1:                   # deterministic fast path
            return outcomes[0]
        r = self._rng.random()
        cumulative = 0.0
        for o in outcomes:
            cumulative += o.prob
            if r < cumulative:
                return o
        return outcomes[-1]                      # rounding safety net