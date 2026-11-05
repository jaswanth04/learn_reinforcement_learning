from typing import Protocol, TypeVar, Any

from utils.models import TrainingParameters, ValueFunction, AgentParameters

T = TypeVar("T", bound=TrainingParameters, contravariant=True)
P = TypeVar("P", bound=AgentParameters, contravariant=True)

class Agent(Protocol[T]):

    @property
    def name(self) -> str: ...

    def train(self, train_params: T) -> ValueFunction: ...

    def predict(self, state: int) -> int: ...


class AgentFactory(Protocol):
    def __call__(self, params: Any) -> Agent[Any]: ...