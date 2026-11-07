"""In-memory store of environments, keyed by a stable env_id.

Each registration is parsed into a typed environment model
(registry/env_models.py) using the `env_type` sent with the payload, and the
model is built into a live environment. Environments are deduplicated by the
model's *structural* key (the fields that define the MDP; seed is ignored) so
equivalent environments -- e.g. re-initialized with a different seed -- share
the same env_id. Models (experiments) reference an env_id, which lets the
front-end only offer models whose environment matches the one currently
configured, preventing e.g. loading a 16-state model into a 9-state lake.
"""

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from exceptions.registry_execeptions import RegistryException
from registry.env_models import EnvironmentModel, model_for
from core.models import Environment


@dataclass
class StoredEnvironment:
    """A registered environment: its typed model plus the built env object."""

    model: EnvironmentModel
    env: Environment


class EnvironmentRegistry:
    """Runtime store of environments keyed by env_id.

    Deduplication is structural: registering the same MDP twice (e.g. with a
    different seed) yields the same env_id.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, StoredEnvironment] = {}
        self._by_key: dict[str, str] = {}

    def get_or_create(
        self,
        env_config: dict[str, Any],
        env_type: str | None = None,
    ) -> str:
        """Return the env_id for `env_config`, reusing an equivalent one.

        `env_type` (as sent by the front-end) selects the model; it may also
        be carried inside `env_config` itself.
        """
        env_type = env_type or env_config.get("env_type")
        model = model_for(env_type).from_json(env_config)
        key = model.structural_key()

        existing = self._by_key.get(key)
        if existing is not None:
            return existing

        env_id = str(uuid4())
        self._by_id[env_id] = StoredEnvironment(model=model, env=model.build())
        self._by_key[key] = env_id
        return env_id

    def get_model(self, env_id: str) -> EnvironmentModel:
        """Return the typed model stored for `env_id`."""
        return self._stored(env_id).model

    def get_env(self, env_id: str) -> Environment:
        """Return the live environment built for `env_id`."""
        return self._stored(env_id).env

    def get(self, env_id: str) -> dict[str, Any]:
        """Return the full config dict for `env_id` (model fields plus the
        derived P table, actions, skip_states, ...) as consumed by the front-end."""
        stored = self._stored(env_id)
        return stored.model.to_dict(stored.env)

    def list(self) -> dict[str, dict[str, Any]]:
        """Return {env_id: config} for every stored environment."""
        return {
            env_id: stored.model.to_dict(stored.env)
            for env_id, stored in self._by_id.items()
        }

    def _stored(self, env_id: str) -> StoredEnvironment:
        if env_id not in self._by_id:
            raise RegistryException(
                f"Environment {env_id} not found in the registry"
            )
        return self._by_id[env_id]

    def has(self, env_id: str) -> bool:
        return env_id in self._by_id

    def __contains__(self, env_id: str) -> bool:
        return env_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)
