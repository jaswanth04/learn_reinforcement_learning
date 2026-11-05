from dataclasses import dataclass
from typing import Any

from agents.base import Agent
from utils.models import TrainingParameters, ValueFunction


@dataclass
class ModelCard:
    """Snapshot of an experiment, ready to store or inspect."""
    experiment_id: str
    experiment_name: str
    env_type: str
    agent_type: str
    agent_params: dict[str, Any]
    training_params: dict[str, Any]
    trained: bool
    value_function: ValueFunction | None = None
    env_id: str = ""

@dataclass(eq=False)
class Experiment:
    experiment_id: str
    experiment_name: str
    env_type: str
    agent_type: str
    agent: Agent[Any]                              # ready, already-initialized agent
    agent_params: dict[str, Any]
    env_id: str = ""
    training_parameter_class: type[TrainingParameters] | None = None

    # state
    training_params: dict[str, Any] | None = None
    trained: bool = False
    value_function: ValueFunction | None = None

    def train(self, train_params: dict[str, Any]) -> ValueFunction:
        """Build validated training params (if a class is registered) and run the agent."""
        if self.training_parameter_class is not None:
            params_obj = self.training_parameter_class.from_dict(train_params)
        else:
            params_obj = train_params

        V = self.agent.train(params_obj)

        self.trained = True
        self.value_function = V
        self.training_params = dict(train_params)
        return V

    def predict(self, state: int) -> int:
        """Return the greedy action for the given state from the trained agent."""
        return self.agent.predict(state)

    def model_card(self) -> ModelCard:
        return ModelCard(
            experiment_id=self.experiment_id,
            experiment_name=self.experiment_name,
            env_type=self.env_type,
            agent_type=self.agent_type,
            agent_params=dict(self.agent_params),
            training_params=dict(self.training_params or {}),
            trained=self.trained,
            value_function=self.value_function,
            env_id=self.env_id,
        )

