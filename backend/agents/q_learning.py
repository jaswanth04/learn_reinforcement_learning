import numpy as np

from core.models import ValueFunction

from utils.policy_control.exploration import epsilon_greedy, greedy_action

from agents.td import TemporalDifferenceAgentParameters, TemporalDifferenceAgent



class QLearningAgent(TemporalDifferenceAgent):
    def __init__(self, params: TemporalDifferenceAgentParameters) -> None:
        super().__init__(params)

    def train_by_step(self, current_state: int, action_taken: int, reward: float, next_state: int, done: bool) -> tuple[int, ValueFunction]:

        self.check_train_parameters_initialized()

        alpha = self.current_alpha
        epsilon = self.current_epsilon

        next_action = greedy_action(next_state, self.Q) if done else epsilon_greedy(next_state, self.Q, epsilon)

        td_target = reward if done else reward + self.gamma * np.max(self.Q[next_state])

        td_error = td_target - self.Q[current_state, action_taken]
        self.Q[current_state, action_taken] += alpha * td_error

        self.update_time_step()

        return (next_action, self.value_function())



class DoubleQLearningAgent(TemporalDifferenceAgent):
    def __init__(self, params: TemporalDifferenceAgentParameters) -> None:
        super().__init__(params)

        self.Q1 = np.zeros((params.n_states, params.n_actions))
        self.Q2 = np.zeros((params.n_states, params.n_actions))

    def train_by_step(self, current_state: int, action_taken: int, reward: float, next_state: int, done: bool) -> tuple[int, ValueFunction]:
        self.check_train_parameters_initialized()

        alpha = self.current_alpha
        epsilon = self.current_epsilon

        # Pick the greedy next action with one table, evaluate it with the other
        if np.random.randint(2):
            argmax_Q1 = np.argmax(self.Q1[next_state])
            td_target = reward if done else reward + self.gamma * self.Q2[next_state, argmax_Q1]
            td_error = td_target - self.Q1[current_state, action_taken]
            self.Q1[current_state, action_taken] += alpha * td_error
        else:
            argmax_Q2 = np.argmax(self.Q2[next_state])
            td_target = reward if done else reward + self.gamma * self.Q1[next_state, argmax_Q2]
            td_error = td_target - self.Q2[current_state, action_taken]
            self.Q2[current_state, action_taken] += alpha * td_error

        self.Q = (self.Q1 + self.Q2) / 2
        next_action = greedy_action(next_state, self.Q) if done else epsilon_greedy(next_state, self.Q, epsilon)

        self.update_time_step()

        return (next_action, self.value_function())





