"""Policy control (generalized policy iteration): learn Q(s, a) and a greedy policy."""

from .eligibility_traces import q_lambda, sarsa_lambda
from .exploration import epsilon_greedy, greedy_action
from .monte_carlo import mc_control
from .online_planning import OnlineDynaQ, OnlinePlanningAgent, OnlineTrajectorySampling
from .planning import dyna_q, trajectory_sampling
from .temporal_difference import double_q_learning, q_learning, sarsa

__all__ = [
    "OnlineDynaQ",
    "OnlinePlanningAgent",
    "OnlineTrajectorySampling",
    "double_q_learning",
    "dyna_q",
    "epsilon_greedy",
    "greedy_action",
    "mc_control",
    "q_lambda",
    "q_learning",
    "sarsa",
    "sarsa_lambda",
    "trajectory_sampling",
]
