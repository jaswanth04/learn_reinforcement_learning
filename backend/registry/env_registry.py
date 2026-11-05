"""In-memory store of environment configurations, keyed by a stable env_id.

Environments are deduplicated by a *structural* fingerprint (n_states, hole /
goal positions, slip prob, rewards, start, ...) so equivalent environments -- e.g.
re-initialized walks that differ only in seed -- share the same env_id. Models
(experiments) reference an env_id, which lets the front-end only offer models
whose environment matches the one currently configured, preventing e.g. loading
a 10-state model into a 6-state corridor.
"""

import json
from typing import Any
from uuid import uuid4

from exceptions.registry_execeptions import RegistryException

# P (transition table) and seed do not change which model fits an environment:
# P is derived from the other fields and seed only drives RNG noise.
_IGNORED_KEYS = {"P", "seed"}


def _as_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_as_jsonable(o) for o in obj]
    if isinstance(obj, dict):
        return {str(k): _as_jsonable(v) for k, v in obj.items()}
    return str(obj)


def env_fingerprint(env_config: dict[str, Any]) -> str:
    """Deterministic *structural* fingerprint of an env_config dict."""
    structural = {
        k: v for k, v in env_config.items() if k not in _IGNORED_KEYS
    }
    return json.dumps(_as_jsonable(structural), sort_keys=True, default=str)


class EnvironmentRegistry:
    """Runtime store of environment configs keyed by env_id."""

    def __init__(self) -> None:
        self._by_id: dict[str, dict[str, Any]] = {}
        self._by_fingerprint: dict[str, str] = {}

    def get_or_create(self, env_config: dict[str, Any]) -> str:
        """Return the env_id for `env_config`, reusing an equivalent one."""
        fp = env_fingerprint(env_config)
        existing = self._by_fingerprint.get(fp)
        if existing is not None:
            stored = self._by_id[existing]
            # Fill in any richer detail (e.g. the P table) a later, structurally
            # identical registration carries, so the stored env stays complete.
            for key, value in env_config.items():
                if value and not stored.get(key):
                    stored[key] = value
            return existing

        env_id = str(uuid4())
        self._by_id[env_id] = dict(env_config)
        self._by_fingerprint[fp] = env_id
        return env_id

    def get(self, env_id: str) -> dict[str, Any]:
        """Return a copy of the stored env config for `env_id`."""
        if env_id not in self._by_id:
            raise RegistryException(
                f"Environment {env_id} not found in the registry"
            )
        return dict(self._by_id[env_id])

    def list(self) -> dict[str, dict[str, Any]]:
        """Return {env_id: config} for every stored environment."""
        return {env_id: dict(cfg) for env_id, cfg in self._by_id.items()}

    def has(self, env_id: str) -> bool:
        return env_id in self._by_id

    def __contains__(self, env_id: str) -> bool:
        return env_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)
