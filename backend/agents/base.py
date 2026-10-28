from typing import Protocol, Any
from utils.models import ValueFunction

class Agent(Protocol):

    @property
    def name() -> str: ...

    def train(self, train_params: dict[str, Any]) -> ValueFunction: ...

    def predict(self, state: int) -> int: ...