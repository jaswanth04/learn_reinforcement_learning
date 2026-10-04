from dataclasses import dataclass
from typing import Callable, Protocol

from numpy.typing import NDArray
import numpy as np


@dataclass
class ActionResult:
    """One outcome branch of a transition, OR a realized sample from step()."""
    prob: float | None      # None when step() has already sampled
    next_state: int
    reward: float
    done: bool


@dataclass
class Action:
    name: str
    code: int
    symbol: str


type ProbabilityTransition = dict[int, dict[int, list[ActionResult]]]
type ActionMap = dict[int, Action]
type Policy = Callable[[int], int]              # state -> action code
type ValueFunction = NDArray[np.float64]        # V[s]
type QTable = NDArray[np.float64]               # Q[s, a]


class Environment(Protocol):
    """Implemented by BanditWalk, BanditSlipperyWalk and FrozenLake."""

    @property
    def P(self) -> ProbabilityTransition: ...
    @property
    def actions(self) -> ActionMap: ...
    @property
    def n_states(self) -> int: ...
    @property
    def display_cols(self) -> int: ...
    @property
    def goal(self) -> int: ...
    @property
    def holes(self) -> list[int]: ...
    @property
    def state(self) -> int: ...

    def reset(self) -> int: ...
    def step(self, action: int | Action | str) -> ActionResult: ...
    def seed(self, seed: int | None = None) -> None: ...
    def display(self, show_indices: bool = True, show_legend: bool = False) -> None: ...