
from dataclasses import dataclass, field, fields, MISSING
from typing import Callable, Any, Self

from agents.base import get_init_policy, decay_schedule
from core.models import ValueFunction
from utils.policy_control.exploration import epsilon_greedy, greedy_action
from exceptions.agent_exceptions import AgentInitializationException, AgentTrainingException
import numpy as np

from numpy.typing import NDArray
from abc import ABC, abstractmethod

@dataclass
class OnlinePlanningAgentParameters:
    name: str
    env_type: str
    n_states: int
    n_actions: int
    initial_policy: Callable[[int], int] | None = None

    def set_policy(self, policy_type: str | None = None):

        self.initial_policy = get_init_policy(self.n_actions, policy_type)

    @classmethod
    def from_dict(cls, agent_params: dict[str, Any]) -> Self:
        known_fields = {f.name for f in fields(cls)}

        missing = [f.name for f in fields(cls)
                   if f.name not in agent_params
                   and f.default is MISSING
                   and f.default_factory is MISSING]
        if missing:
            raise AgentInitializationException(
                f"Missing required agent parameters: {', '.join(missing)}"
            )

        filtered = {k: v for k, v in agent_params.items() if k in known_fields}
        # initial_policy arrives as a plain string name over the wire (JSON);
        # resolve it to a callable here so the agent stays policy-agnostic.
        if isinstance(filtered.get("initial_policy"), str):
            filtered["initial_policy"] = get_init_policy(
                int(filtered["n_actions"]), filtered["initial_policy"]
            )
        return cls(**filtered)


@dataclass
class OnlinePlanningTrainingParameters:
    initial_state: int
    # gamma < 1 on purpose: Dyna-Q replays zero-reward self-loop transitions
    # (wall bumps) from its model, and with gamma = 1 those updates are
    # non-contraction — Q inflates, the greedy policy can lock onto the
    # "stay put" action and bulk prediction collapses (Sutton & Barto Ex. 8.4).
    gamma: float = 0.95
    init_alpha: float = 0.5
    min_alpha: float = 0.01
    alpha_decay_ratio: float = 0.5
    init_epsilon: float = 1.0
    min_epsilon: float = 0.1
    epsilon_decay_ratio: float = 0.9
    n_episodes: int = 1000
    max_steps: int = 200
    # 30 planning rollouts per real step (Sutton & Barto Fig. 8.4): with the
    # conservative gamma/alpha above this is what keeps the learned model's
    # optimism from inflating Q on zero-reward self-loops.
    n_planning: int = 30
    max_trajectory_depth: int = 100

    alphas: NDArray[np.float64] | None = field(default=None, repr=False)
    epsilons: NDArray[np.float64] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.alphas is None:          # don't silently overwrite user-supplied arrays
            self.alphas = decay_schedule(
                self.init_alpha, self.min_alpha, self.alpha_decay_ratio, self.n_episodes
            )
        if self.epsilons is None:
            self.epsilons = decay_schedule(
                self.init_epsilon, self.min_epsilon, self.epsilon_decay_ratio, self.n_episodes
            )

    def _at(self, schedule: NDArray[np.float64] | None, step: int, name: str) -> float:
        if schedule is None or schedule.size == 0:   # narrows to NDArray[np.float64]
            raise RuntimeError(f"{name} schedule is not initialised")
        # Index the schedules by episode; past the end they keep their final value.
        step = min(max(int(step), 0), schedule.size - 1)
        return float(schedule[step])              # np.float64 -> float, satisfies -> float

    def get_alpha(self, episode: int) -> float:
        return self._at(self.alphas, episode, "alpha")

    def get_epsilon(self, episode: int) -> float:
        return self._at(self.epsilons, episode, "epsilon")

    @classmethod
    def from_dict(cls, train_params: dict[str, Any]) -> Self:
        known_fields = {f.name for f in fields(cls)} - {"alphas", "epsilons"}

        missing = [f.name for f in fields(cls)
                   if f.name in known_fields
                   and f.default is MISSING
                   and f.default_factory is MISSING
                   and f.name not in train_params]
        if missing:
            raise AgentInitializationException(
                f"Missing required training parameters: {', '.join(missing)}"
            )

        filtered = {k: v for k, v in train_params.items() if k in known_fields}
        return cls(**filtered)


class OnlinePlanningAgent(ABC):
    def __init__(self, params: OnlinePlanningAgentParameters) -> None:

        self._trained = False
        self._name = params.name
        self._env_type = params.env_type

        self._n_states = params.n_states
        self._n_actions = params.n_actions

        self._policy = params.initial_policy or get_init_policy(params.n_actions)

        self.Q = np.zeros((self._n_states, self._n_actions))

        self._train_params: OnlinePlanningTrainingParameters | None = None

        self.T_count = np.zeros((self._n_states, self._n_actions, self._n_states))  # Transition counts
        self.R_model = np.zeros((self._n_states, self._n_actions, self._n_states))  # Mean reward per transition
        # (state, action) pairs the model has actually seen. Planning can only
        # simulate those, so this replaces the per-step sums over T_count.
        self.modelled = np.zeros((self._n_states, self._n_actions), dtype=bool)
        self._episode = -1
        self._episode_trained = 0 # Cumulative across all runs on this Q

    def train(self, train_params: OnlinePlanningTrainingParameters | None = None) -> ValueFunction:
        raise AgentTrainingException(
            f"{type(self).__name__} learns from interaction, not from a full MDP. "
            "Drive it with start_training() / start_episode() / train_by_step() "
            "(WebSocket endpoint /ws/train_step)."
        )

    def check_train_parameters_initialized(self):
        if self._train_params is None:
            raise AgentTrainingException("Train parameters are not yet initialized")

    @property
    def name(self) -> str:
        return self._name

    @property
    def episode(self) -> int:
        return self._episode

    @property
    def episodes_trained(self) -> int:
        """Episodes seen by this Q table across ALL training runs."""
        return self._episode_trained

    @property
    def train_step(self) -> int:
        return self._train_step

    @property
    def trained(self) -> bool:
        return self._trained

    @property
    def gamma(self):
        self.check_train_parameters_initialized()
        assert self._train_params is not None
        return self._train_params.gamma

    @property
    def alpha(self):
        self.check_train_parameters_initialized()
        assert self._train_params is not None

        return self._train_params.get_alpha(self._episode)

    @property
    def epsilon(self):
        self.check_train_parameters_initialized()
        assert self._train_params is not None

        return self._train_params.get_epsilon(self._episode)

    @property
    def n_planning(self):
        self.check_train_parameters_initialized()
        assert self._train_params is not None
        return self._train_params.n_planning

    @property
    def max_trajectory_depth(self):
        self.check_train_parameters_initialized()
        assert self._train_params is not None
        return self._train_params.max_trajectory_depth

    current_alpha = alpha
    current_epsilon = epsilon

    
    def update_model(self, state: int, action:int, reward: float, next_state:int) -> None:
        self.T_count[state, action, next_state] += 1
        self.modelled[state, action] = True
        r_diff = reward - self.R_model[state, action, next_state]
        self.R_model[state, action, next_state] += r_diff / self.T_count[state, action, next_state]

    def value_function(self) -> ValueFunction:
        """V(s) = max_a Q(s, a) — the greedy value of the current Q table."""
        return np.max(self.Q, axis=1)

    def update_time_step(self):
        self._train_step += 1

    def update_q(self, reward: float, state: int, action: int, next_state: int, done: bool):
        td_target = reward if done else reward + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error        

    
    @abstractmethod
    def train_by_step(self, current_state: int, action_taken: int, reward: float, next_state: int, done: bool) -> tuple[int, ValueFunction]: ...

    def predict(self, state: int):
        if self._trained:
            return greedy_action(state, self.Q)
        else:
            return self._policy(state)

    def start_training(self, train_params: OnlinePlanningTrainingParameters) -> tuple[int, ValueFunction]:
        self._train_params = train_params
        self._train_step = 0
        self._episode = -1
        self._trained = False

        return self.start_episode()

    def start_episode(self):
        self.check_train_parameters_initialized()
        assert self._train_params is not None

        self._episode += 1
        self._episode_trained += 1

        action = epsilon_greedy(self._train_params.initial_state, self.Q, self.epsilon)

        return (action, self.value_function())

    def finish(self):
        self._trained = True


