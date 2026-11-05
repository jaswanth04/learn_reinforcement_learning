from typing import Any

from agents.base import Agent
from exceptions.registry_execeptions import RegistryException
from registry.agent_registry import AgentRegistry


class AgentInstanceRegistry:
    """Runtime store of *initialized* agents keyed by agent id.

    On a miss, lazily builds a new agent from the type catalog (AgentRegistry).
    """

    def __init__(self, agent_registry: AgentRegistry):
        self._agent_registry = agent_registry
        self._instances: dict[str, Agent[Any]] = {}

    def get_or_create(
        self,
        agent_id: str,
        agent_type: str,
        agent_params: dict[str, Any],
    ) -> Agent[Any]:
        """Return the agent with `agent_id`, building it if not present."""
        existing = self._instances.get(agent_id)
        if existing is not None:
            return existing

        entry = self._agent_registry.get_agent(agent_type)
        params = dict(agent_params)

        if entry.agent_parameter_class is not None:
            init_params = entry.agent_parameter_class.from_dict(params)
            agent: Agent[Any] = entry.agent_class(init_params)
        else:
            agent = entry.agent_class(params)

        self._instances[agent_id] = agent
        return agent

    def get(self, agent_id: str) -> Agent[Any]:
        if agent_id not in self._instances:
            raise RegistryException(f"Agent {agent_id} not found in instance registry")
        return self._instances[agent_id]

    def has(self, agent_id: str) -> bool:
        return agent_id in self._instances

    def remove(self, agent_id: str) -> None:
        self._instances.pop(agent_id, None)

    def get_agent_ids(self) -> list[str]:
        return list(self._instances.keys())

    def clear(self) -> None:
        self._instances.clear()