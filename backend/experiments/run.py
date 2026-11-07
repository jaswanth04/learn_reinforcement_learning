from dataclasses import dataclass
from typing import Any

from agents.base import Agent
from core.models import TrainingParameters, ValueFunction


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
    episodes_trained: int | None = None

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

    def _build_params(self, train_params: dict[str, Any]) -> Any:
        if self.training_parameter_class is not None:
            return self.training_parameter_class.from_dict(train_params)
        return train_params

    def train(self, train_params: dict[str, Any]) -> ValueFunction:
        """Build validated training params (if a class is registered) and run the agent."""
        params_obj = self._build_params(train_params)

        V = self.agent.train(params_obj)

        self.trained = True
        self.value_function = V
        self.training_params = dict(train_params)
        return V

    # ------------------------------------------------ step-wise (online) ---

    def _stepwise_agent(self) -> Any:
        if not all(hasattr(self.agent, m) for m in ("train_by_step", "finish")):
            raise TypeError(
                f"Agent type {self.agent_type!r} does not support step-wise training"
            )
        return self.agent

    def start_training(self, train_params: dict[str, Any]) -> tuple[int, ValueFunction]:
        """Begin a fresh interaction-based training run (e.g. SARSA).

        Returns the first action of episode 0 and the current value function.
        The run is only partially trained until finish_training() is called.
        """
        params_obj = self._build_params(train_params)
        self.training_params = dict(train_params)
        self.trained = False

        action, V = self._stepwise_agent().start_training(params_obj)
        self.value_function = V
        return action, V

    def next_episode(self) -> tuple[int, ValueFunction]:
        """Start the next episode of the running training run."""
        action, V = self._stepwise_agent().start_episode()
        self.value_function = V
        return action, V

    def train_step(self, current_state: int, action_taken: int, reward: float,
                   next_state: int, done: bool) -> tuple[int, ValueFunction]:
        """Apply one observed transition; returns the next action + value function.

        Keeps the experiment *untrained*: only finish_training() marks it
        trained, so an interrupted run is never reported as fully trained.
        """
        action, V = self._stepwise_agent().train_by_step(
            current_state, action_taken, reward, next_state, done
        )
        self.value_function = V
        return action, V

    def finish_training(self) -> None:
        """Mark the step-wise training run as complete (fully trained)."""
        self._stepwise_agent().finish()
        self.trained = True

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
            episodes_trained=getattr(self.agent, "episodes_trained", None),
        )

