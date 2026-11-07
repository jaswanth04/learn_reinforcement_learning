"""Typed environment models: one dataclass per environment kind.

The front-end sends `env_type` together with a config payload. Each model
here knows how to:

  - `from_json(payload)` : parse that payload (ignoring client-only metadata
    such as `name`, `preset`, `actions`, `skip_states` and `P`),
  - `build()`           : construct the live environment (the env classes in
    backend/utils), so the backend owns the transition table instead of
    trusting a client-computed one,
  - `structural_key()`  : a deterministic key over the fields that actually
    define the MDP (seed is excluded: it only drives RNG noise), used by the
    registry to dedupe equivalent environments into one env_id,
  - `to_dict(env)`      : render the full config back out for the front-end,
    including the derived P / actions / skip_states / n_states.

Adding a new environment = add one dataclass below and register it in
ENV_MODELS.
"""

import json
import types
from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, ClassVar, Self

from exceptions.registry_execeptions import RegistryException
from core.models import ActionResult, Environment, EnvironmentType
from utils.bandit_slippery_walk import BanditSlipperyWalk
from utils.frozen_lake import FrozenLake


def _coerce(value: Any, typ: Any) -> Any:
    """Best-effort coercion of a JSON value to a field's declared type
    (JSON numbers arrive as int or float regardless of the target type)."""
    if value is None or typ is None:
        return value
    if isinstance(typ, types.GenericAlias) and typ.__origin__ is list:
        item_type = typ.__args__[0]
        return [_coerce(v, item_type) for v in value]
    if typ is int and not isinstance(value, bool):
        return int(value)
    if typ is float:
        return float(value)
    if typ is str:
        return str(value)
    return value


def transitions_to_json(P: dict[int, dict[int, list[ActionResult]]]) -> dict[str, Any]:
    """Render an env's P table in the JSON shape the front-end consumes:
    P[s][a] = [{prob, next_state, reward, done}, ...]."""
    return {
        str(s): {
            str(a): [
                {"prob": o.prob, "next_state": o.next_state,
                 "reward": o.reward, "done": o.done}
                for o in outcomes
            ]
            for a, outcomes in actions.items()
        }
        for s, actions in P.items()
    }


@dataclass(frozen=True)
class EnvironmentModel:
    """Base class for one environment model per env_type.

    Subclasses declare exactly the fields that define the MDP. `env_type` is
    a ClassVar (not a field) so it never pollutes the structural key.
    """

    env_type: ClassVar[str]

    # Fields that don't change the MDP, excluded from the structural key so
    # equivalent environments (e.g. same layout, different seed) dedupe.
    volatile_fields: ClassVar[frozenset[str]] = frozenset({"seed"})

    # ------------------------------------------------------------------ parsing
    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Self:
        """Parse a front-end config payload into this model.

        Unknown keys are ignored (client metadata); missing keys fall back to
        the field's default when it has one.
        """
        data: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in payload:
                data[f.name] = _coerce(payload[f.name], f.type)

        missing = [
            f.name for f in fields(cls)
            if f.name not in data
            and f.default is MISSING
            and f.default_factory is MISSING
        ]
        if missing:
            raise RegistryException(
                f"Environment {cls.env_type} is missing required field(s): "
                f"{', '.join(missing)}"
            )
        return cls(**data)

    # ------------------------------------------------------------------ identity
    def structural_key(self) -> str:
        """Deterministic key over the MDP-defining fields (seed excluded)."""
        data = {
            k: v for k, v in asdict(self).items()
            if k not in self.volatile_fields
        }
        return json.dumps(data, sort_keys=True)

    # ------------------------------------------------------------------ building
    def build(self) -> Environment:
        """Construct the live environment for this configuration."""
        raise NotImplementedError

    # ------------------------------------------------------------------ rendering
    def derived_fields(self, env: Environment) -> dict[str, Any]:
        """Front-end-facing fields computed from the built env."""
        return {
            "n_actions": len(env.actions),
            "actions": {a.name: a.code for a in env.actions.values()},
            "skip_states": [*env.holes, env.goal],
            "P": transitions_to_json(env.P),
        }

    def to_dict(self, env: Environment | None = None) -> dict[str, Any]:
        """Full config dict for the front-end: model fields + derived fields."""
        env = env if env is not None else self.build()
        cfg: dict[str, Any] = asdict(self)
        cfg["env_type"] = self.env_type
        cfg.update(self.derived_fields(env))
        return cfg


@dataclass(frozen=True)
class SlipperyBanditWalkModel(EnvironmentModel):
    """1-D corridor with one hole; moves slip with probability slip_prob."""

    env_type: ClassVar[str] = EnvironmentType.SlipperyBanditWalk.value

    n_states: int
    start: int
    hole: int
    goal: int
    slip_prob: float = 0.2
    goal_reward: float = 1.0
    hole_reward: float = 0.0
    step_reward: float = 0.0
    seed: int = 0

    def build(self) -> Environment:
        return BanditSlipperyWalk(
            n_states=self.n_states,
            start=self.start,
            hole=self.hole,
            goal=self.goal,
            slip_prob=self.slip_prob,
            goal_reward=self.goal_reward,
            hole_reward=self.hole_reward,
            step_reward=self.step_reward,
            seed=self.seed,
        )


@dataclass(frozen=True)
class FrozenLakeModel(EnvironmentModel):
    """n x n slippery gridworld; slip_probs is (left, intended, right)."""

    env_type: ClassVar[str] = EnvironmentType.FrozenLake.value

    n: int
    start: int
    goal: int
    holes: list[int]
    slip_probs: list[float] = field(
        default_factory=lambda: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    )
    goal_reward: float = 1.0
    hole_reward: float = 0.0
    step_reward: float = 0.0
    seed: int = 0

    def build(self) -> Environment:
        return FrozenLake(
            n=self.n,
            holes=list(self.holes),
            start=self.start,
            goal=self.goal,
            slippery=True,
            slip_probs=tuple(self.slip_probs),
            goal_reward=self.goal_reward,
            hole_reward=self.hole_reward,
            step_reward=self.step_reward,
            seed=self.seed,
        )

    def derived_fields(self, env: Environment) -> dict[str, Any]:
        cfg = super().derived_fields(env)
        cfg["n_states"] = env.n_states
        cfg["intended_prob"] = self.slip_probs[1]
        return cfg


# env_type string (as sent by the front-end) -> model class.
ENV_MODELS: dict[str, type[EnvironmentModel]] = {
    model.env_type: model
    for model in (SlipperyBanditWalkModel, FrozenLakeModel)
}


def model_for(env_type: Any) -> type[EnvironmentModel]:
    """Resolve the model class for an env_type string; RegistryException if unknown."""
    try:
        return ENV_MODELS[env_type]
    except (KeyError, TypeError):
        known = ", ".join(sorted(ENV_MODELS))
        raise RegistryException(
            f"Unknown environment type: {env_type!r}. Known types: {known}"
        ) from None
