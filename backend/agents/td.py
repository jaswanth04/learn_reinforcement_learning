import numpy as np
from typing import Callable, Any, Self
import random
from dataclasses import MISSING, dataclass, fields, field
from exceptions.agent_exceptions import AgentInitializationException, AgentTrainingException
from core.models import ValueFunction
from utils.policy_control.exploration import epsilon_greedy, greedy_action
from numpy.typing import NDArray

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


@dataclass
class SarsaAgentParameters:
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
class SarsaAgentTrainingParameters:
    initial_state: int
    gamma: float = 1.0
    init_alpha: float = 0.5
    min_alpha: float = 0.5
    alpha_decay_ratio: float = 0.5
    init_epsilon: float = 1.0
    min_epsilon: float = 0.1
    epsilon_decay_ratio: float = 0.9
    n_episodes: int = 1000
    max_steps: int = 200

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


class SarsaAgent:
    """On-policy TD control (SARSA).

    The agent does not need the MDP: the environment is stepped by the
    caller (the front end). A training run is driven through three calls:

      1. start_training(params)  -> first (action, V) of episode 0
      2. train_by_step(s, a, r, s', done) -> next (action, V)
      3. start_episode()         -> first (action, V) of the next episode

    alpha / epsilon follow schedules indexed by the *episode* number.
    """

    def __init__(self, params: SarsaAgentParameters) -> None:
        self._name = params.name
        self._env_type = params.env_type
        self.Q = np.zeros((params.n_states, params.n_actions), dtype=np.float64)
        self._n_states = params.n_states
        self._n_actions = params.n_actions

        self._policy = params.initial_policy or get_init_policy(params.n_actions)
        self._trained = False

        self._train_params: SarsaAgentTrainingParameters | None = None
        self._train_step: int = 0
        self._episode: int = -1        # no episode started yet (this run)
        self._episodes_trained: int = 0  # cumulative across ALL runs on this Q table

    @property
    def name(self):
        return self._name

    @property
    def episode(self) -> int:
        return self._episode

    @property
    def episodes_trained(self) -> int:
        """Episodes seen by this Q table across ALL training runs.

        start_training() keeps the Q values (warm start), so this cumulative
        count is what tells you how much total experience the model has.
        """
        return self._episodes_trained

    @property
    def train_step(self) -> int:
        return self._train_step

    @property
    def current_alpha(self) -> float | None:
        if self._train_params is None:
            return None
        return self._train_params.get_alpha(self._episode)

    @property
    def current_epsilon(self) -> float | None:
        if self._train_params is None:
            return None
        return self._train_params.get_epsilon(self._episode)

    @property
    def trained(self) -> bool:
        return self._trained

    def set_training_params(self, train_params: SarsaAgentTrainingParameters):
        self._train_params = train_params

    def value_function(self) -> ValueFunction:
        """V(s) = max_a Q(s, a) — the greedy value of the current Q table."""
        return np.max(self.Q, axis=1)

    def predict(self, state: int):
        if self._trained:
            return greedy_action(state, self.Q)
        else:
            return self._policy(state)

    def train(self, train_params: SarsaAgentTrainingParameters) -> ValueFunction:
        raise AgentTrainingException(
            "SarsaAgent learns from interaction, not from a full MDP. "
            "Drive it with start_training() / start_episode() / train_by_step() "
            "(WebSocket endpoint /ws/train_step)."
        )

    def start_training(self, train_params: SarsaAgentTrainingParameters) -> tuple[int, ValueFunction]:
        """Begin a fresh training run (resets step/episode counters and the
        trained flag) and take the first action of episode 0."""
        self._train_params = train_params
        self._train_step = 0
        self._episode = -1
        self._trained = False
        return self.start_episode()

    def start_episode(self) -> tuple[int, ValueFunction]:
        """Start the next episode: epsilon-greedy action from the initial state."""
        if self._train_params is None:
            raise AgentTrainingException("Train parameters are not yet initialized")

        self._episode += 1
        self._episodes_trained += 1
        epsilon = self._train_params.get_epsilon(self._episode)
        action = epsilon_greedy(self._train_params.initial_state, self.Q, epsilon)

        return (action, self.value_function())

    def train_by_step(self, current_state: int, action_taken: int, reward: float, next_state: int, done: bool) -> tuple[int, ValueFunction]:

        if self._train_params is None:
            raise AgentTrainingException("Train parameters are not yet initialized")

        alpha = self._train_params.get_alpha(self._episode)
        epsilon = self._train_params.get_epsilon(self._episode)

        if done:
            td_target = reward
            next_action = greedy_action(next_state, self.Q)
        else:
            next_action = epsilon_greedy(next_state, self.Q, epsilon)
            td_target = reward + self._train_params.gamma * self.Q[next_state, next_action]

        td_error = td_target - self.Q[current_state, action_taken]
        self.Q[current_state, action_taken] += alpha * td_error

        self._train_step += 1

        return (next_action, self.value_function())

    def finish(self) -> None:
        """Mark the training run as complete: from now on predict() returns
        the greedy policy over Q. A run that never calls finish() stays
        partially trained (and is reported as not trained)."""
        self._trained = True
