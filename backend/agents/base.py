from typing import Protocol, TypeVar, Any

from core.models import TrainingParameters, ValueFunction, AgentParameters

T = TypeVar("T", bound=TrainingParameters, contravariant=True)
P = TypeVar("P", bound=AgentParameters, contravariant=True)

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
