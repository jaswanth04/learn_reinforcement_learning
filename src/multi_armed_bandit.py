"""Multi-Armed Bandit: a SINGLE-state environment.

Pull an arm -> get a reward sample. There is no next state, no dynamics,
and no goal/hole -- it intentionally is NOT an MDP (that is the whole
point of the bandit setting).

Each pull counts as one episode/trial: step() returns done=True, so
  env.reset() ; r = env.step(a).reward
is one complete trial, matching the episodic loop used in the bandit
learning agents (greedy / epsilon-greedy / optimistic / UCB).
"""

from env_core import EnvCore
from models import Action, ActionResult


def bernoulli(p: float):
    """Reward sampler: 1.0 with prob p, else 0.0."""
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    return lambda rng: 1.0 if rng.random() < p else 0.0


def gaussian(mean: float, std: float):
    """Reward sampler: N(mean, std)."""
    mean, std = float(mean), float(std)
    if std < 0:
        raise ValueError("std must be >= 0")
    return lambda rng: float(rng.gauss(mean, std))


class MultiArmedBandit(EnvCore):

    def __init__(self, n_arms=10, reward_probs=None, reward_dists=None, seed=None):
        """Arms are Bernoulli by default (reward_probs), or arbitrary
        callables via reward_dists: sample(rng) -> float.

        Examples
        --------
        env = MultiArmedBandit(n_arms=5, seed=0)                  # random probs
        env = MultiArmedBandit(reward_probs=[0.2, 0.8], seed=0)    # 2 arms
        env = MultiArmedBandit(reward_dists=[gaussian(0, 1), gaussian(3, 1)],
                               seed=0)
        """
        super().__init__(seed=seed)
        self.n_arms = int(n_arms)
        if self.n_arms < 2:
            raise ValueError("a bandit needs at least 2 arms")

        if reward_dists is not None:
            if len(reward_dists) != self.n_arms:
                raise ValueError(
                    f"reward_dists has {len(reward_dists)} samplers but n_arms={self.n_arms}")
            self._samplers = list(reward_dists)
            self._true_values: list[float | None] = [None] * self.n_arms
        else:
            if reward_probs is None:
                # Reproducible random Bernoulli arms (seeded).
                reward_probs = [round(self._rng.random(), 2) for _ in range(self.n_arms)]
            if len(reward_probs) != self.n_arms:
                raise ValueError(
                    f"reward_probs has {len(reward_probs)} probs but n_arms={self.n_arms}")
            probs = [float(p) for p in reward_probs]
            if any(not 0.0 <= p <= 1.0 for p in probs):
                raise ValueError("reward_probs must all be in [0, 1]")
            self._samplers = [bernoulli(p) for p in probs]
            self._true_values = list(probs)          # Bernoulli mean == p

        self._build_action_map()
        self.start_state = 0                         # single state
        self.reset()

    # ------------------------------------------------------------- setup
    def _build_action_map(self):
        self._actions: dict[int, Action] = {
            i: Action(name=f"Arm {i}", code=i, symbol=str(i))
            for i in range(self.n_arms)
        }
        self._actions_by_name = {a.name: a.code for a in self._actions.values()}

    # --------------------------------------------------------- interaction
    def step(self, action: int | Action | str) -> ActionResult:
        """Pull an arm and sample its reward.

        Single state, so next_state is always 0 and done is True: one pull
        is one completed trial/episode.
        """
        code = self._resolve_action(action)
        if code not in self._actions:
            raise ValueError(f"invalid arm {action!r} (have 0..{self.n_arms - 1})")

        reward = self._samplers[code](self._rng)
        self._state = 0
        return ActionResult(prob=None, next_state=0, reward=float(reward), done=True)

    # ------------------------------------------------------------ display
    def display(self, show_indices: bool = True, show_legend: bool = True) -> None:
        """Show the arms and their TRUE expected rewards.

        (The agent never sees these -- only the samples. Displaying them
        here is for YOUR eyes to check that learning converges correctly.)
        """
        if show_legend:
            print("Legend: rows = arm name / true mean reward (agent can't see it)")

        print("| " + " |".join(f"{i:^5}" for i in range(self.n_arms)) + " |")
        if show_indices:
            vals = []
            for tv in self._true_values:
                vals.append(f"{tv:5.2f}" if tv is not None else "  ?  ")
            print("| " + " |".join(vals) + " |")

    # ------------------------------------------------------- read-only api
    @property
    def P(self):
        raise AttributeError(
            "MultiArmedBandit has no transition table P -- it is a bandit, "
            "not an MDP. Use true_values for the arm means instead.")

    @property
    def actions(self):
        return self._actions

    @property
    def state(self):
        return self._state

    @property
    def n_states(self):
        return 1

    @property
    def display_cols(self):
        return self.n_arms

    @property
    def true_values(self) -> list[float | None]:
        """Expected reward of each arm (None when reward_dists were given)."""
        return list(self._true_values)