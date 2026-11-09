
from core.models import ValueFunction

from utils.policy_control.exploration import epsilon_greedy, greedy_action

from agents.td import TemporalDifferenceAgentParameters, TemporalDifferenceAgent


class SarsaAgent(TemporalDifferenceAgent):
    def __init__(self, params: TemporalDifferenceAgentParameters) -> None:
        super().__init__(params)

    def train_by_step(self, current_state: int, action_taken: int, reward: float, next_state: int, done: bool) -> tuple[int, ValueFunction]:

        self.check_train_parameters_initialized()

        alpha = self.current_alpha
        epsilon = self.current_epsilon

        if done:
            td_target = reward
            next_action = greedy_action(next_state, self.Q)
        else:
            next_action = epsilon_greedy(next_state, self.Q, epsilon)
            td_target = reward + self.gamma * self.Q[next_state, next_action]

        td_error = td_target - self.Q[current_state, action_taken]

        # Updating E
        self.update_E(state=current_state, action=action_taken)

        self.Q += alpha * td_error * self.E
        self.E *= self.gamma * self.lambda_

        self.update_time_step()

        return (next_action, self.value_function())
