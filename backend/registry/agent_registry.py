from agents.base import Agent
from exceptions.registry_execeptions import RegistryException

class AgentRegistry:
    def __init__(self):
        self._registry: dict[str, Agent] = {}

    def register(self, agent_type: str, agent: Agent) -> None:
        print(f"Registering {agent_type}")
        self._registry[agent_type] = agent

    def get_agent(self, agent_type: str) -> Agent:
        if agent_type not in self._registry:
            raise RegistryException(f"Agent type: {agent_type} is not registered")

        return self._registry[agent_type]

    def get_agent_types(self) -> list[str]:
        return list(self._registry.keys())