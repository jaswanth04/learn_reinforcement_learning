"""RussellNorvigGridWorld: the classic 3x4 gridworld from AIMA (Russell & Norvig).

Standard layout (book 1-based coordinates, row 1 on top):

    .  .  .  +1      s0  s1  s2  s3
    .  #  .  -1      s4  s5  s6  s7
    S  .  .  .       s8  s9  s10 s11

i.e. start s8 ((3,1)), impassable wall s5 ((2,2)), +1 goal s3 ((1,4)),
-1 pit s7 ((2,4)) -- 0-indexed row-major state ids on a 3x4 board.

DIFFERENCES FROM FrozenLake:
  - the board is rectangular (rows x cols), not square;
  - the wall is IMPASSABLE (a move into it or off the grid leaves the agent
    in place), not a terminal hole;
  - there are TWO terminal states: +1 goal and -1 pit, absorbing with their
    reward counted on entry (V stays 0 on the terminals themselves, same
    convention as FrozenLake's goal);
  - the default living reward is -0.04 per step (FrozenLake: 0).

SLIP SEMANTICS (action-code space, same convention as FrozenLake):
when the agent picks action a, the ACTUAL move is drawn from
    [ (a-1)%4 ,  a ,  (a+1)%4 ]
with slip_probs (left, intended, right) = (0.1, 0.8, 0.1) by default.
Moves into the wall or off the grid are merged into a stay-put branch.

`holes` (Environment protocol) exposes [wall, pit]: the states DP must treat
as non-evaluated / absorbing, which is exactly right -- the wall is
unreachable and the pit's -1 is counted on entry.
"""

from utils.env_core import EnvCore
from core.models import Action, ActionResult

# Action codes (Gymnasium convention, same as FrozenLake)
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

# (drow, dcol) per action
_DELTAS = {
    LEFT:  (0, -1),
    DOWN:  (1,  0),
    RIGHT: (0,  1),
    UP:    (-1, 0),
}

# (prob left of intended, prob intended, prob right of intended)
_DEFAULT_SLIP = (0.1, 0.8, 0.1)

# Standard AIMA 3x4 layout as 0-indexed row-major state ids.
_ROWS, _COLS = 3, 4
_START = 8          # (3,1)
_WALL = 5           # (2,2)
_GOAL = 3           # (1,4)  +1
_PIT = 7            # (2,4)  -1


class RussellNorvigGridWorld(EnvCore):

    def __init__(self, rows=_ROWS, cols=_COLS, walls=None, start=_START,
                 goal=_GOAL, pit=_PIT, slip_probs=_DEFAULT_SLIP,
                 goal_reward=1.0, pit_reward=-1.0, step_reward=-0.04,
                 seed=None):
        if rows < 1 or cols < 1:
            raise ValueError("grid rows and cols must be >= 1")
        super().__init__(seed=seed)

        self.rows = int(rows)
        self.cols = int(cols)
        self.n_states = self.rows * self.cols

        self.walls = list(walls) if walls is not None else [_WALL]
        self.start_state = int(start)
        self.pos_goal = int(goal)
        self.pit = int(pit)

        self.slip_probs = tuple(float(p) for p in slip_probs)
        self.goal_reward = float(goal_reward)
        self.pit_reward = float(pit_reward)
        self.step_reward = float(step_reward)

        self._verify()
        self._build_action_map()
        self._build_transitions()
        self.reset()

    # ------------------------------------------------------------ configuration
    def _verify(self):
        n = self.n_states
        specials = {"start": self.start_state, "goal": self.pos_goal, "pit": self.pit}
        for name, s in specials.items():
            if not 0 <= s < n:
                raise ValueError(f"{name} must be in 0..{n-1}, got {s}")
        for w in self.walls:
            if not 0 <= w < n:
                raise ValueError(f"wall must be in 0..{n-1}, got {w}")
        if len(set(self.walls)) != len(self.walls):
            raise ValueError("walls must be unique")
        distinct = [self.start_state, self.pos_goal, self.pit, *self.walls]
        if len(set(distinct)) != len(distinct):
            raise ValueError("start, goal, pit and walls must all differ")

        if len(self.slip_probs) != 3:
            raise ValueError("slip_probs must be (left, intended, right)")
        if any(p < 0 for p in self.slip_probs):
            raise ValueError("slip_probs must be non-negative")
        if abs(sum(self.slip_probs) - 1.0) > 1e-9:
            raise ValueError(f"slip_probs must sum to 1, got {sum(self.slip_probs)}")

        self._walls_set = set(self.walls)

    def _build_action_map(self):
        self._actions: dict[int, Action] = {
            LEFT:  Action("LEFT",  LEFT,  "<"),
            DOWN:  Action("DOWN",  DOWN,  "v"),
            RIGHT: Action("RIGHT", RIGHT, ">"),
            UP:    Action("UP",    UP,    "^"),
        }
        self._actions_by_name = {a.name: a.code for a in self._actions.values()}

    def _row(self, s: int) -> int:
        return s // self.cols

    def _col(self, s: int) -> int:
        return s % self.cols

    def _action_distribution(self, code: int) -> tuple[list[int], list[float]]:
        """Return (actual-move codes, probabilities) for a chosen action.

        Slippery => [ (a-1)%4, a, (a+1)%4 ] with slip_probs
        (left, intended, right in action-code space, FrozenLake convention).
        """
        p_left, p_intended, p_right = self.slip_probs
        return ([(code - 1) % 4, code, (code + 1) % 4],
                [p_left, p_intended, p_right])

    # ------------------------------------------------------------ transitions
    def _build_transitions(self):
        P = {}
        for s in range(self.n_states):
            P[s] = {}
            for code in self._actions:
                if s == self.pos_goal or s == self.pit:
                    # Absorbing terminal: every branch stays and ends the episode.
                    P[s][code] = [ActionResult(1.0, s, self.step_reward, True)]
                    continue
                if s in self._walls_set:
                    # Fake unreachable cell: inert self-loop (never stepped into).
                    P[s][code] = [ActionResult(1.0, s, self.step_reward, True)]
                    continue

                move_codes, probs = self._action_distribution(code)

                # Merge branches that land on the same (state, reward, done).
                branches: dict[tuple, float] = {}
                for prob, c in zip(probs, move_codes):
                    if prob <= 0.0:
                        continue
                    drow, dcol = _DELTAS[c]
                    nrow = self._row(s) + drow
                    ncol = self._col(s) + dcol
                    if not (0 <= nrow < self.rows and 0 <= ncol < self.cols):
                        nxt = s                                   # off-grid: stay
                    else:
                        cand = nrow * self.cols + ncol
                        # Wall: impassable, the agent stays in place.
                        nxt = cand if cand not in self._walls_set else s

                    if nxt == self.pos_goal:
                        reward, done = self.goal_reward, True
                    elif nxt == self.pit:
                        reward, done = self.pit_reward, True
                    else:
                        reward, done = self.step_reward, False

                    key = (nxt, reward, done)
                    branches[key] = branches.get(key, 0.0) + prob

                P[s][code] = [ActionResult(p, nxt, r, d)
                              for (nxt, r, d), p in branches.items()]
        self._P = P

    # ---------------------------------------------------------------- display
    def display(self, show_indices: bool = True, show_legend: bool = True) -> None:
        """2D board: S=start, .=open, #=wall, +=goal(+1), -=pit(-1), A=agent."""
        if show_legend:
            print("Legend: S=start  .=open  #=wall  +=goal(+1)  -=pit(-1)  A=agent")

        def cell(s: int) -> str:
            if s == self.pos_goal:
                marker = "+"
            elif s == self.pit:
                marker = "-"
            elif s in self._walls_set:
                marker = "#"
            elif s == self.start_state:
                marker = "S"
            else:
                marker = "."
            return f"{marker}{'A' if s == self._state else ' ':<4}"

        for r in range(self.rows):
            row = range(r * self.cols, (r + 1) * self.cols)
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
    def goal(self):
        """The +1 terminal (protocol: single goal). The -1 pit lives in `holes`."""
        return self.pos_goal

    @property
    def holes(self):
        """States DP must treat as absorbing/un-evaluated: the wall and the pit."""
        return [*self.walls, self.pit]

    @property
    def display_cols(self):
        return self.cols
