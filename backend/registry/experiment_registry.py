from typing import Any

from exceptions.registry_execeptions import RegistryException


class ExperimentRegistry:
    """In-memory store mapping experiment ids to Experiment instances."""

    def __init__(self):
        self._registry: dict[str, Any] = {}

    def store_experiment(self, id: str, experiment: Any) -> None:
        self._registry[id] = experiment

    def retrieve(self, id: str) -> Any:
        if id not in self._registry:
            raise RegistryException(f"Experiment {id} not found in the registry")
        return self._registry[id]

    def has_experiment(self, id: str) -> bool:
        return id in self._registry

    def get_model_ids(self) -> list[str]:
        return list(self._registry.keys())

    def clear(self) -> None:
        self._registry.clear()

    def __contains__(self, id: str) -> bool:
        return id in self._registry
