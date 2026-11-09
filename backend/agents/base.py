from typing import Protocol, TypeVar, Any, Callable

from core.models import TrainingParameters, ValueFunction, AgentParameters

from numpy.typing import NDArray
import numpy as np
import random
from exceptions.agent_exceptions import AgentInitializationException

T = TypeVar("T", bound=TrainingParameters, contravariant=True)
P = TypeVar("P", bound=AgentParameters, contravariant=True)

def decay_schedule(init_value: float,
                   min_value: float,
                   decay_ratio: float,
                   max_steps: int,
                   log_start: int = -2,
                   log_base: float = 10) -> NDArray[np.float64]:
    decay_steps = max(min(int(max_steps * decay_ratio), max_steps), 2)
    rem_steps = max_steps - decay_steps
    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    return np.pad(values, (0, rem_steps), "edge")


def get_init_policy(n_actions: int, policy_type: str | None = None) -> Callable[[Any], int]:

    policy = policy_type if policy_type else "random"
    policy_map: dict[str, Callable[[int], int]] = {
        "random": lambda _: random.choice(range(n_actions)),
        "always_left": lambda _: 0,
        "always_right": lambda _: 1
        }
    if policy not in policy_map:
        raise AgentInitializationException(
            f"Unknown initial policy: {policy!r}. Known policies: {', '.join(policy_map)}"
        )
    return policy_map[policy]


class Agent(Protocol[T]):

    @property
    def name(self) -> str: ...

    def train(self, train_params: T) -> ValueFunction: ...

    def predict(self, state: int) -> int: ...


class StepwiseAgent(Agent[T], Protocol[T]):
    """An Agent that learns from interaction (e.g. SARSA) instead of solving
    the full MDP in train().

    The caller owns the environment: start_training() opens a run, each
    observed transition is reported through train_by_step(), start_episode()
    resets to the initial state, and finish() marks the run as COMPLETE.
    Until finish() is called the agent is only partially trained: it must
    not be reported as trained, and predict() keeps using the behaviour
    policy.
    """

    @property
    def episode(self) -> int: ...

    @property
    def episodes_trained(self) -> int:
        """Episodes seen across ALL runs on this agent's Q table (warm
        starts included)."""
        ...

    @property
    def train_step(self) -> int: ...

    @property
    def trained(self) -> bool: ...

    def start_training(self, train_params: T) -> tuple[int, ValueFunction]: ...

    def start_episode(self) -> tuple[int, ValueFunction]: ...

    def train_by_step(self, current_state: int, action_taken: int, reward: float,
                      next_state: int, done: bool) -> tuple[int, ValueFunction]: ...

    def finish(self) -> None: ...


class AgentFactory(Protocol):
    def __call__(self, params: Any) -> Agent[Any]: ...
