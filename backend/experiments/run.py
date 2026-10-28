from typing import Any

from registry.agent_registry import AgentRegistry
from agents.policy_iteration import PolicyIterationAgent
from agents.base import Agent

from utils.models import ValueFunction, ActionResult


from uuid import uuid4

agent_registry = AgentRegistry()

agent_registry.register("PolicyIteration", PolicyIterationAgent)


def normalize_transitions(P: dict[str, Any]) -> dict[int, dict[int, list[ActionResult]]]:
    """Convert a JSON-shaped P table (string keys, dict outcomes) into the
    int-keyed ActionResult form consumed by the agents."""
    return {
        int(s): {
            int(a): [
                ActionResult(
                    prob=float(o.get("prob")),
                    next_state=int(o.get("next_state")),
                    reward=float(o.get("reward")),
                    done=bool(o.get("done")),
                )
                for o in outcomes
            ]
            for a, outcomes in actions.items()
        }
        for s, actions in P.items()
    }


class Experiment:
    def __init__(
        self,
        experiment_name: str,
        agent_type: str,
        agent_params: dict[str, Any],
        env_config: dict[str, Any] | None = None,
    ):
        agent = agent_registry.get_agent(agent_type)

        self.env_config = dict(env_config or {})

        params = dict(agent_params or {})
        params.setdefault("name", experiment_name)
        params.setdefault("env_type", self.env_config.get("name", "generic"))
        params.setdefault("n_states", self.env_config["n_states"])
        params.setdefault("n_actions", self.env_config["n_actions"])

        self._initialized_agent: Agent = agent(**params)
        self.name = experiment_name
        self.id = str(uuid4())
        self.trained = False
        self.value_function: ValueFunction | None = None
        self.policy: list[int] | None = None

    def train(self, train_params: dict[str, Any]) -> ValueFunction:
        params = dict(train_params or {})
        params["P"] = normalize_transitions(self.env_config["P"])
        params.setdefault("skip_states", self.env_config.get("skip_states", []))

        V = self._initialized_agent.train(params)
        self.trained = True
        self.value_function = V
        self.policy = [
            int(self._initialized_agent.predict(s))
            for s in range(int(self.env_config["n_states"]))
        ]
        return V

    def predict(self, state: int) -> int:
        return self._initialized_agent.predict(state)
