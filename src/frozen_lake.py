"""FrozenLake: a configurable, slippery gridworld.

Classic 4x4 board (default):

    SFFF
    FHFH
    FFFH
    HFFG

i.e. start 0, goal 15, holes at 5, 7, 11, 12 (0-indexed row-major).

SLIP SEMANTICS (same as Gymnasium FrozenLake-v1):
when the agent picks action a, the ACTUAL move is drawn from
    [ (a-1)%4 ,  a ,  (a+1)%4 ]
each with probability 1/3.  So the intended direction lands 1/3 of the
time, the direction "left of it" in action-number space 1/3 of the time,
and "right of it" 1/3 of the time.  Moves that walk off the board are
clamped, so the agent stays put (that stay-branch is merged with any
real outcome that lands on the same state).
"""

from env_core import EnvCore
from models import Action, ActionResult

# Action codes (Gymnasium convention)
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

# (drow, dcol) per action
_DELTAS = {
    LEFT:  (0, -1),
    DOWN:  (1,  0),
    RIGHT: (0,  1),
    UP:    (-1, 0),
}

# (prob left of intended, prob intended, prob right of intended)
_DEFAULT_SLIP = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


class FrozenLake(EnvCore):

    def __init__(self, n=4, holes=None, start=0, goal=None, slippery=True,
                 slip_probs=_DEFAULT_SLIP,
                 goal_reward=1.0, hole_reward=0.0, step_reward=0.0,
                 seed=None):
        if n < 2:
            raise ValueError("grid size n must be >= 2")
        super().__init__(seed=seed)

        self.n = int(n)
        self.n_states = self.n * self.n

        # The classic 4x4 holes: 5, 7, 11, 12.
        self.holes = list(holes) if holes is not None else [5, 7, 11, 12]
        self.start_state = int(start)
        self.goal = int(goal) if goal is not None else self.n_states - 1

        self.goal_reward = float(goal_reward)
        self.hole_reward = float(hole_reward)
        self.step_reward = float(step_reward)
        self.slippery = bool(slippery)
        self.slip_probs = tuple(float(p) for p in slip_probs)

        self._verify()
        self._build_action_map()
        self._build_transitions()
        self.reset()

    # ------------------------------------------------------------ configuration
    def _verify(self):
        n = self.n_states
        if not 0 <= self.start_state < n:
            raise ValueError(f"start must be in 0..{n-1}, got {self.start_state}")
        if not 0 <= self.goal < n:
            raise ValueError(f"goal must be in 0..{n-1}, got {self.goal}")
        for h in self.holes:
            if not 0 <= h < n:
                raise ValueError(f"hole must be in 0..{n-1}, got {h}")
        if len(set(self.holes)) != len(self.holes):
            raise ValueError("holes must be unique")
        if self.goal in self.holes:
            raise ValueError("goal must not be a hole")
        if self.start_state in self.holes:
            raise ValueError("start must not be a hole")
        if self.start_state == self.goal:
            raise ValueError("start must differ from goal")

        if self.slippery:
            if len(self.slip_probs) != 3:
                raise ValueError("slip_probs must be (left, intended, right)")
            if any(p < 0 for p in self.slip_probs):
                raise ValueError("slip_probs must be non-negative")
            if abs(sum(self.slip_probs) - 1.0) > 1e-9:
                raise ValueError(f"slip_probs must sum to 1, got {sum(self.slip_probs)}")

        self._holes_set = set(self.holes)

    def _build_action_map(self):
        self._actions: dict[int, Action] = {
            LEFT:  Action("LEFT",  LEFT,  "<"),
            DOWN:  Action("DOWN",  DOWN,  "v"),
            RIGHT: Action("RIGHT", RIGHT, ">"),
            UP:    Action("UP",    UP,    "^"),
        }
        self._actions_by_name = {a.name: a.code for a in self._actions.values()}

    def _row(self, s: int) -> int:
        return s // self.n

    def _col(self, s: int) -> int:
        return s % self.n

    def _action_distribution(self, code: int) -> tuple[list[int], list[float]]:
        """Return (actual-move codes, probabilities) for a chosen action.

        Slippery => [ (a-1)%4, a, (a+1)%4 ] with slip_probs.
        Not slippery => the intended action, always.
        """
        if not self.slippery:
            return [code], [1.0]
        p_left, p_intended, p_right = self.slip_probs
        return ([(code - 1) % 4, code, (code + 1) % 4],
                [p_left, p_intended, p_right])

    # ------------------------------------------------------------ transitions
    def _build_transitions(self):
        P = {}
        for s in range(self.n_states):
            P[s] = {}
            for code in self._actions:
                if s == self.goal or s in self._holes_set:
                    # Absorbing terminal: every branch stays and ends the episode.
                    P[s][code] = [ActionResult(1.0, s, self.step_reward, True)]
                    continue

                move_codes, probs = self._action_distribution(code)

                # Merge branches that land on the same (state, reward, done).
                branches: dict[tuple, float] = {}
                for prob, c in zip(probs, move_codes):
                    if prob <= 0.0:
                        continue
                    drow, dcol = _DELTAS[c]
                    nrow = min(max(self._row(s) + drow, 0), self.n - 1)  # wall clamp
                    ncol = min(max(self._col(s) + dcol, 0), self.n - 1)
                    nxt = nrow * self.n + ncol

                    if nxt == self.goal:
                        reward, done = self.goal_reward, True
                    elif nxt in self._holes_set:
                        reward, done = self.hole_reward, True
                    else:
                        reward, done = self.step_reward, False

                    key = (nxt, reward, done)
                    branches[key] = branches.get(key, 0.0) + prob

                P[s][code] = [ActionResult(p, nxt, r, d)
                              for (nxt, r, d), p in branches.items()]
        self._P = P

    # ---------------------------------------------------------------- display
    def display(self, show_indices: bool = True, show_legend: bool = True) -> None:
        """2D board: S=start, F=frozen, H=hole, G=goal, A=agent."""
        if show_legend:
            print("Legend: S=start  F=frozen  H=hole  G=goal  A=agent")

        def cell(s: int) -> str:
            if s == self.goal:
                marker = "G"
            elif s in self._holes_set:
                marker = "H"
            elif s == self.start_state:
                marker = "S"
            else:
                marker = "F"
            return f"{marker}{'A' if s == self._state else ' ':<4}"

        for r in range(self.n):
            row = range(r * self.n, (r + 1) * self.n)
            print("| " + " |".join(cell(s) for s in row) + " |")
            if show_indices:
                print("| " + " |".join(f"{s:^5}" for s in row) + " |")

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
    def display_cols(self):
        return self.n