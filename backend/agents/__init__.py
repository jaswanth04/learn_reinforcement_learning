"""Training agents and their parameter dataclasses.

This package is the public entry point for agents: import the classes you
need from `agents` (e.g. `from agents import QLearningAgent`) instead of
reaching into the individual modules.

Conventions every agent follows:
  * model-based agents (dynamic programming) implement train(train_params);
  * interaction-based agents (TD control, Dyna-style planning) implement
    start_training() / start_episode() / train_by_step() / finish() and raise
    from train() — see StepwiseAgent.
"""

from agents.base import Agent, AgentFactory, StepwiseAgent
from agents.policy_iteration import (
    PolicyIterationAgent,
    PolicyIterationAgentParameters,
    PolicyIterationTrainingParameters,
)
from agents.td import (
    TemporalDifferenceAgent,
    TemporalDifferenceAgentParameters,
    TemporalDifferenceTrainingParameters,
)
from agents.sarsa import SarsaAgent
from agents.q_learning import DoubleQLearningAgent, QLearningAgent
from agents.online_planning import (
    OnlinePlanningAgent,
    OnlinePlanningAgentParameters,
    OnlinePlanningTrainingParameters,
)
from agents.dyna_q import DynaQAgent
from agents.trajectory_sampling import TrajectorySamplingAgent

__all__ = [
    "Agent",
    "AgentFactory",
    "StepwiseAgent",
    "PolicyIterationAgent",
    "PolicyIterationAgentParameters",
    "PolicyIterationTrainingParameters",
    "TemporalDifferenceAgent",
    "TemporalDifferenceAgentParameters",
    "TemporalDifferenceTrainingParameters",
    "SarsaAgent",
    "QLearningAgent",
    "DoubleQLearningAgent",
    "OnlinePlanningAgent",
    "OnlinePlanningAgentParameters",
    "OnlinePlanningTrainingParameters",
    "DynaQAgent",
    "TrajectorySamplingAgent",
]
