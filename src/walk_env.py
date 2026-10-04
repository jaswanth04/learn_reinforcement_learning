"""Shared logic for the 1D walks (deterministic or slippery)."""

from env_core import EnvCore
from models import Action, ActionResult


class WalkEnv(EnvCore):
    """Subclasses implement `_distribution(s, code)` for non-terminal states."""

    def __init__(self, n_states=3, start=None, hole=None, goal=None,
                 goal_reward=1.0, hole_reward=0.0, step_reward=0.0, seed=None):
        if n_states < 3:
            raise ValueError("n_states must be >= 3")
        super().__init__(seed=seed)

        self.n_states = int(n_states)
        self.hole = int(hole) if hole is not None else 0
        self.goal = int(goal) if goal is not None else self.n_states - 1
        self.start_state = int(start) if start is not None else self.n_states // 2
        self.goal_reward = float(goal_reward)
        self.hole_reward = float(hole_reward)
        self.step_reward = float(step_reward)

        self._verify()
        self._build_action_map()
        self._build_transitions()
        self.reset()

    # ------------------------------------------------------------ configuration
    def _verify(self):
        for name, v in (("hole", self.hole), ("goal", self.goal)):
            if not 0 <= v < self.n_states:
                raise ValueError(f"{name} must be in 0..{self.n_states-1}, got {v}")
        if self.hole == self.goal:
            raise ValueError(f"goal ({self.goal}) must differ from hole ({self.hole})")
        if not 0 <= self.start_state < self.n_states:
            raise ValueError(
                f"start_state must be in 0..{self.n_states-1}, got {self.start_state}")
        if self.start_state in (self.hole, self.goal):
            raise ValueError(
                f"start_state ({self.start_state}) must not be the hole or the goal")

    def _build_action_map(self):
        self._actions: dict[int, Action] = {
            0: Action("LEFT", 0, "<"),
            1: Action("RIGHT", 1, ">"),
        }
        self._actions_by_name = {a.name: a.code for a in self._actions.values()}

    def _build_transitions(self):
        P = {}
        for s in range(self.n_states):
            P[s] = {}
            for a in self._actions:
                if s in (self.hole, self.goal):
                    P[s][a] = [ActionResult(1.0, s, self.step_reward, True)]
                else:
                    P[s][a] = self._distribution(s, a)
        self._P = P

    def _distribution(self, s: int, code: int) -> list[ActionResult]:
        raise NotImplementedError("subclasses must implement _distribution")

    def _check_wall_bounce(self, s: int) -> int:
        # Write wall bounce clearly
        nxt = s

        if s < 0:
            nxt = 0

        if s >= self.n_states:
            nxt = self.n_states - 1

        return nxt

    # ---------------------------------------------------------------- display
    def display(self, show_indices: bool = True, show_legend: bool = False) -> None:
        """1D corridor: G=goal H=hole .=frozen A=agent."""
        n = self.n_states

        if show_legend:
            print("Legend: G=goal  H=hole  .=frozen  A=agent")

        def cell(s: int) -> str:
            marker = "G" if s == self.goal else "H" if s == self.hole else "."
            return f"{marker}{'A' if s == self._state else ' ':<4}"

        print("| " + " |".join(cell(s) for s in range(n)) + " |")
        if show_indices:
            print("| " + " |".join(f"{s:^5}" for s in range(n)) + " |")

    # ------------------------------------------------------------ read-only api
    @property
    def P(self):
        return self._P

    @property
    def actions(self):
        return self._actions

    @property
    def state(self):
        return self._state

    @property
    def holes(self) -> list[int]:
        return [self.hole]

    @property
    def display_cols(self):
        return self.n_states