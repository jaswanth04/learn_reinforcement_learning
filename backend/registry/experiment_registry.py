from experiments.run import Experiment, ModelCard
from exceptions.registry_execeptions import RegistryException


class ExperimentRegistry:
    """In-memory store keyed by experiment_id. Stores Experiments; exposes
    ModelCards (snapshots) for inspection and filtering."""

    def __init__(self):
        self._registry: dict[str, Experiment] = {}

    # --- registration / storage ---

    def register_experiment(self, experiment: Experiment) -> None:
        self._registry[experiment.experiment_id] = experiment

    def store_experiment(self, experiment_id: str, experiment: Experiment) -> None:
        # kept as a compatibility alias; prefer register_experiment
        self._registry[experiment_id] = experiment

    # --- lookup ---

    def retrieve(self, experiment_id: str) -> Experiment:
        if experiment_id not in self._registry:
            raise RegistryException(f"Experiment {experiment_id} not found in the registry")
        return self._registry[experiment_id]

    def has_experiment(self, experiment_id: str) -> bool:
        return experiment_id in self._registry

    def get_model_ids(self) -> list[str]:
        return list(self._registry.keys())

    # --- model cards ---

    def get_model_card(self, experiment_id: str) -> ModelCard:
        return self.retrieve(experiment_id).model_card()

    def get_model_cards(self) -> list[ModelCard]:
        return [e.model_card() for e in self._registry.values()]

    # --- filtering ---

    def filter(
        self,
        env_type: str | None = None,
        agent_type: str | None = None,
        trained: bool | None = None,
        env_id: str | None = None,
    ) -> list[ModelCard]:
        """Return model cards matching all provided filters (AND semantics)."""
        cards = self.get_model_cards()
        if env_type is not None:
            cards = [c for c in cards if c.env_type == env_type]
        if agent_type is not None:
            cards = [c for c in cards if c.agent_type == agent_type]
        if trained is not None:
            cards = [c for c in cards if c.trained == trained]
        if env_id is not None:
            cards = [c for c in cards if c.env_id == env_id]
        return cards

    def filter_by_env_type(self, env_type: str) -> list[ModelCard]:
        return [c for c in self.get_model_cards() if c.env_type == env_type]

    def filter_by_agent_type(self, agent_type: str) -> list[ModelCard]:
        return [c for c in self.get_model_cards() if c.agent_type == agent_type]

    # --- lifecycle ---

    def clear(self) -> None:
        self._registry.clear()

    def __contains__(self, experiment_id: str) -> bool:
        return experiment_id in self._registry

    def __len__(self) -> int:
        return len(self._registry)