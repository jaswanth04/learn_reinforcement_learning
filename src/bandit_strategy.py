"""
Multi-arm bandit experiment harness with a strategy registry.

The shared loop (array allocation, episode loop, incremental Q/N update,
result packaging) lives in ``get_bandit_results``. Strategies are registered
by name and resolved to an *initialiser* (builds Q/N) plus an *action
selector* (picks the arm at each trial), so calling code only needs a name:

    results = get_bandit_results(env, "epsilon_greedy", n_episodes=2000, epsilon=0.05)

Everything is typed via the ``Initializer``/``ActionSelector`` Protocols, and
new strategies can be added without touching the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from models import Environment


@dataclass
class BanditEpisodicResults:
    returns: np.ndarray
    Q_e: np.ndarray
    actions: np.ndarray
    episodes: int
    title: str


# ---------------------------------------------------------------------------
# Pluggable callable contracts
# ---------------------------------------------------------------------------

class Initializer(Protocol):
    """Build (Q, N) from the number of arms plus strategy kwargs."""

    def __call__(self, n_arms: int, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        ...


class ActionSelector(Protocol):
    """Pick one arm given Q, N, the current trial index, and strategy kwargs."""

    def __call__(
        self,
        Q: np.ndarray,
        N: np.ndarray,
        step: int,
        n_episodes: int,
        **kwargs: Any,
    ) -> int:
        ...


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategySpec:
    """A named recipe: how to initialise plus how to select actions."""

    name: str
    init: Initializer
    select: ActionSelector
    title: str
    defaults: Mapping[str, Any] = field(default_factory=dict)


_REGISTRY: dict[str, StrategySpec] = {}


def _canonical(name: str) -> str:
    """Registry key: lowercased, with '_' and '-' collapsed to spaces."""
    return name.strip().lower().replace("_", " ").replace("-", " ")


def _default_name(selector: ActionSelector) -> str:
    raw = getattr(selector, "__name__", "unnamed")
    return raw[7:] if raw.startswith("select_") else raw


def register_strategy(
    init: Initializer,
    select: ActionSelector,
    *,
    name: str | None = None,
    title: str | None = None,
    defaults: Mapping[str, Any] | None = None,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
) -> StrategySpec:
    """Register a strategy so it can be invoked by name in ``get_bandit_results``.

    ``defaults`` become the strategy's hyperparameters; any call-site kwargs
    with the same keys override them.
    """
    if name is None:
        name = _default_name(select)
    if title is None:
        title = name.replace("_", " ").replace("-", " ").title()

    spec = StrategySpec(
        name=name,
        init=init,
        select=select,
        title=title,
        defaults=dict(defaults or {}),
    )

    keys = (_canonical(name), *(_canonical(a) for a in aliases))
    for key in keys:
        if key in _REGISTRY and not replace:
            raise ValueError(
                f"Strategy {name!r} already registered (use replace=True to override)."
            )
        _REGISTRY[key] = spec
    return spec


def strategy(
    selector: ActionSelector | None = None,
    *,
    init: Initializer | None = None,
    name: str | None = None,
    title: str | None = None,
    defaults: Mapping[str, Any] | None = None,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
) -> Callable[[ActionSelector], ActionSelector] | ActionSelector:
    """Decorator that registers a selector function as a named strategy.

    Usage::

        @strategy(init=init_q_n, defaults={"epsilon": 0.05})
        def select_my_bandit(Q, N, step, n_episodes, **kwargs):
            ...
    """
    def _wrap(sel: ActionSelector) -> ActionSelector:
        register_strategy(
            init if init is not None else init_q_n,
            sel,
            name=name,
            title=title,
            defaults=defaults,
            aliases=aliases,
            replace=replace,
        )
        return sel

    if selector is not None:      # used as bare decorator: @strategy
        return _wrap(selector)
    return _wrap                 # used as @strategy(...)


def resolve_strategy(
    strategy: str | StrategySpec | tuple[Initializer, ActionSelector],
) -> StrategySpec:
    """Normalise a name, spec, or ``(init, select)`` pair into a ``StrategySpec``."""
    if isinstance(strategy, StrategySpec):
        return strategy
    if isinstance(strategy, str):
        try:
            return _REGISTRY[_canonical(strategy)]
        except KeyError:
            available = ", ".join(list_strategies())
            raise KeyError(
                f"Unknown strategy {strategy!r}. Registered: {available}"
            ) from None
    if isinstance(strategy, tuple) and len(strategy) == 2:
        init, select = strategy
        return StrategySpec(name="custom", init=init, select=select, title="Custom")
    raise TypeError(
        f"strategy must be a name, StrategySpec, or (init, select) tuple; "
        f"got {type(strategy).__name__}"
    )


def list_strategies() -> list[str]:
    """Registered strategy display names (registration order)."""
    return list(dict.fromkeys(spec.name for spec in _REGISTRY.values()))


# ---------------------------------------------------------------------------
# Initialisers
# ---------------------------------------------------------------------------

def init_q_n(
    n_arms: int,
    q_init: float = 0.0,
    n_init: float = 0.0,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Generic initialiser: ``Q = q_init``, ``N = n_init`` for every arm.

    * plain zeros:      ``init_q_n(n_arms)``
    * optimistic start: ``init_q_n(n_arms, q_init=1.0, n_init=10)``
    * Thompson prior:   ``init_q_n(n_arms, n_init=1.0)``
    """
    Q = np.full(n_arms, q_init, dtype=float)
    N = np.full(n_arms, n_init, dtype=float)
    return Q, N


# ---------------------------------------------------------------------------
# Action selectors (strategies)
# ---------------------------------------------------------------------------

def select_greedy(Q: np.ndarray, N: np.ndarray, step: int, n_episodes: int,
                  **kwargs: Any) -> int:
    """Always take the arm with the highest current estimate."""
    return int(np.argmax(Q))


def select_uniform(Q: np.ndarray, N: np.ndarray, step: int, n_episodes: int,
                   **kwargs: Any) -> int:
    """Pick an arm uniformly at random."""
    return int(np.random.randint(len(Q)))


def select_epsilon_greedy(Q: np.ndarray, N: np.ndarray, step: int,
                          n_episodes: int, epsilon: float = 0.01,
                          **kwargs: Any) -> int:
    """Greedy with probability ``1 - epsilon``, otherwise random."""
    if np.random.random() > epsilon:
        return int(np.argmax(Q))
    return int(np.random.randint(len(Q)))


@lru_cache(maxsize=None)
def _decay_schedule(
    init_epsilon: float, min_epsilon: float, n_episodes: int, decay: float
) -> np.ndarray:
    """Log-spaced epsilon schedule (computed once and cached)."""
    decay_episodes = int(n_episodes * decay)
    eps = np.logspace(np.log10(init_epsilon), np.log10(min_epsilon), decay_episodes)
    return np.pad(eps, (0, n_episodes - decay_episodes), mode="edge")


def select_decay_epsilon_greedy(Q: np.ndarray, N: np.ndarray, step: int,
                                n_episodes: int, init_epsilon: float = 1.0,
                                decay: float = 0.1, min_epsilon: float = 0.01,
                                **kwargs: Any) -> int:
    """Epsilon-greedy with a log-spaced decaying epsilon over the run."""
    eps = _decay_schedule(init_epsilon, min_epsilon, n_episodes, decay)
    if np.random.random() > eps[step]:
        return int(np.argmax(Q))
    return int(np.random.randint(len(Q)))


def select_softmax(Q: np.ndarray, N: np.ndarray, step: int, n_episodes: int,
                   init_temp: float = 1.0, decay: float = 0.04,
                   min_temp: float = 0.01, **kwargs: Any) -> int:
    """Sample an arm from a softmax (Boltzmann) distribution over ``Q``."""
    decay_episodes = int(n_episodes * decay)
    temp = (1.0 - step / decay_episodes) * (init_temp - min_temp) + min_temp
    temp = float(np.clip(temp, min_temp, init_temp))

    scaled = Q / temp
    probs = np.exp(scaled - scaled.max())     # numerically stable softmax
    probs = probs / probs.sum()
    return int(np.random.choice(len(Q), p=probs))


def select_upper_confidence_bound(Q: np.ndarray, N: np.ndarray, step: int,
                                  n_episodes: int, c: float = 2.0,
                                  **kwargs: Any) -> int:
    """UCB1: play any never-tried arm first, then maximise Q + c·sqrt(ln t / N)."""
    unplayed = N == 0
    if np.any(unplayed):
        return int(np.flatnonzero(unplayed)[0])
    t = step + 1                              # 1-based trial -> ln t defined
    upper = Q + c * np.sqrt(np.log(t) / N)
    return int(np.argmax(upper))


def select_thompson_sampling(Q: np.ndarray, N: np.ndarray, step: int,
                             n_episodes: int, alpha: float = 1.0,
                             beta: float = 0.0, **kwargs: Any) -> int:
    """One posterior draw per arm, take the argmax (Gaussian bandit)."""
    scale = alpha / np.sqrt(N) + beta
    return int(np.argmax(np.random.normal(loc=Q, scale=scale)))


# ---------------------------------------------------------------------------
# Register the built-in strategies
# ---------------------------------------------------------------------------

register_strategy(init_q_n, select_greedy, name="greedy",
                  title="Exploitation",
                  aliases=("exploitation", "pure_exploitation"))
register_strategy(init_q_n, select_uniform, name="random",
                  title="Exploration",
                  aliases=("exploration", "pure_exploration", "uniform"))
register_strategy(init_q_n, select_epsilon_greedy, name="epsilon_greedy",
                  title="Epsilon Greedy", defaults={"epsilon": 0.01})
register_strategy(init_q_n, select_decay_epsilon_greedy, name="decay_epsilon_greedy",
                  title="Decay Epsilon",
                  defaults={"init_epsilon": 1.0, "decay": 0.1, "min_epsilon": 0.01})
register_strategy(init_q_n, select_greedy, name="optimistic",
                  title="Optimistic Initialization",
                  defaults={"q_init": 1.0, "n_init": 10},
                  aliases=("optimistic_initialization", "optimistic_init"))
register_strategy(init_q_n, select_softmax, name="softmax",
                  title="Softmax",
                  defaults={"init_temp": 1.0, "decay": 0.04, "min_temp": 0.01})
register_strategy(init_q_n, select_upper_confidence_bound, name="ucb",
                  title="Upper Confidence Bound", defaults={"c": 2.0},
                  aliases=("upper_confidence_bound", "ucb1"))
register_strategy(init_q_n, select_thompson_sampling, name="thompson",
                  title="Thompson Sampling",
                  defaults={"alpha": 1.0, "beta": 0.0, "n_init": 1.0},
                  aliases=("thompson_sampling",))


# ---------------------------------------------------------------------------
# Generic driver
# ---------------------------------------------------------------------------

def get_bandit_results(
    env: Environment,
    strategy: str | StrategySpec | tuple[Initializer, ActionSelector] = "greedy",
    n_episodes: int = 5000,
    *,
    title: str | None = None,
    **kwargs: Any,
) -> BanditEpisodicResults:
    """Run any registered bandit strategy by name.

    Parameters
    ----------
    env:
        The bandit environment (``actions``, ``reset()``, ``step()``).
    strategy:
        Registered strategy name (e.g. ``"greedy"``, ``"ucb"``), a
        ``StrategySpec``, or an explicit ``(init, select)`` pair.
    n_episodes:
        Number of trials / episodes to run.
    title:
        Optional override for the result label (defaults to the strategy's).
    **kwargs:
        Strategy hyperparameters (``epsilon``, ``init_temp``, ``c``, ...).
        These override the strategy's registered defaults.
    """
    spec = resolve_strategy(strategy)
    params = dict(spec.defaults)
    params.update(kwargs)                 # call-site kwargs win over defaults

    n_arms = len(env.actions)
    Q, N = spec.init(n_arms, **params)

    Q_e = np.empty((n_episodes, n_arms))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=int)

    for step in range(n_episodes):
        env.reset()
        action = spec.select(Q, N, step, n_episodes, **params)
        result = env.step(action)

        N[action] += 1
        Q[action] += (result.reward - Q[action]) / N[action]

        Q_e[step] = Q
        returns[step] = result.reward
        actions[step] = action

    return BanditEpisodicResults(returns, Q_e, actions, n_episodes,
                                 title or spec.title)


# ---------------------------------------------------------------------------
# Backwards-compatible wrappers for the old function names
# ---------------------------------------------------------------------------

def pure_exploitation(env: Environment, n_episodes: int = 5000) -> BanditEpisodicResults:
    return get_bandit_results(env, "greedy", n_episodes=n_episodes)


def pure_exploration(env: Environment, n_episodes: int = 5000) -> BanditEpisodicResults:
    return get_bandit_results(env, "random", n_episodes=n_episodes)


def epsilon_greedy(env: Environment, epsilon: float = 0.01,
                   n_episodes: int = 5000) -> BanditEpisodicResults:
    return get_bandit_results(env, "epsilon_greedy", n_episodes=n_episodes,
                              epsilon=epsilon)


def decay_epsilon_greedy(env: Environment, init_epsilon: float = 1.0,
                         decay: float = 0.1, min_epsilon: float = 0.01,
                         n_episodes: int = 5000) -> BanditEpisodicResults:
    return get_bandit_results(env, "decay_epsilon_greedy", n_episodes=n_episodes,
                              init_epsilon=init_epsilon, decay=decay,
                              min_epsilon=min_epsilon)


def optimistic_initialization(env: Environment, optimistic_estimate: float = 1.0,
                              init_n: int = 10,
                              n_episodes: int = 5000) -> BanditEpisodicResults:
    return get_bandit_results(env, "optimistic", n_episodes=n_episodes,
                              q_init=optimistic_estimate, n_init=init_n)


def softmax(env: Environment, init_temp: float = 1.0, decay: float = 0.04,
            min_temp: float = 0.01,
            n_episodes: int = 5000) -> BanditEpisodicResults:
    return get_bandit_results(env, "softmax", n_episodes=n_episodes,
                              init_temp=init_temp, decay=decay, min_temp=min_temp)


def upper_confidence_bound(env: Environment, c: float = 2.0,
                           n_episodes: int = 5000) -> BanditEpisodicResults:
    return get_bandit_results(env, "ucb", n_episodes=n_episodes, c=c)


def thompson_sampling(env: Environment, alpha: float = 1.0, beta: float = 0.0,
                      n_episodes: int = 5000) -> BanditEpisodicResults:
    return get_bandit_results(env, "thompson", n_episodes=n_episodes,
                              alpha=alpha, beta=beta)