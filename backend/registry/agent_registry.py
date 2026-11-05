from dataclasses import dataclass

from agents.base import Agent, AgentFactory
from exceptions.registry_execeptions import RegistryException
from utils.models import TrainingParameters, AgentParameters
from typing import Any

@dataclass
class AgentEntry:
    agent_class: AgentFactory
    training_parameter_class: type[TrainingParameters] | None = None
    agent_parameter_class: type[AgentParameters] | None = None

class AgentRegistry:
    def __init__(self):
        self._registry: dict[str, AgentEntry] = {}

    def register(
        self,
        agent_type: str,
        agent: type[Agent[Any]],
        training_parameters: type[TrainingParameters] | None = None,
        agent_parameters: type[AgentParameters] | None = None,
    ) -> None:
        print(f"Registering {agent_type}")
        self._registry[agent_type] = AgentEntry(
            agent_class=agent,
            training_parameter_class=training_parameters,
            agent_parameter_class=agent_parameters
        )

    def get_agent(self, agent_type: str) -> AgentEntry:
        if agent_type not in self._registry:
            raise RegistryException(f"Agent type: {agent_type} is not registered")
        return self._registry[agent_type]

    def get_agent_types(self) -> list[str]:
        return list(self._registry.keys())