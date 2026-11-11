
import numpy as np

from utils.policy_control.exploration import epsilon_greedy, greedy_action
from agents.online_planning import OnlinePlanningAgent, OnlinePlanningAgentParameters
from core.models import ValueFunction



class DynaQAgent(OnlinePlanningAgent):
    def __init__(self, params: OnlinePlanningAgentParameters) -> None:
        super().__init__(params)

    def train_by_step(self, current_state: int, action_taken: int, reward: float, next_state: int, done: bool) -> tuple[int, ValueFunction]:
    
            # Update the environment model
            self.update_model(current_state, action_taken, reward, next_state)
     
            # Update Q Learning
            self.update_q(reward=reward, 
                          state=current_state,
                          action=action_taken,
                          next_state=next_state,
                          done=done)
     
            # Dyna Planning
            for _ in range(self.n_planning):
     
                # Check if there is atleast one Q updated
                if not self.Q.any():
                    break
     
                visited_states = np.flatnonzero(self.modelled.any(axis=1))
                p_state = int(np.random.choice(visited_states)) # Randomly selecting a visited state
                actions_taken = np.flatnonzero(self.modelled[p_state]) # Filtering all the actions taken till now
                p_action = int(np.random.choice(actions_taken)) # Randomly selecting an action out of actions taken
    
                # Sampling the possible next_state from the model of (p_state, p_action)
                counts = self.T_count[p_state, p_action]
                probs = counts / counts.sum()
                p_next_state = int(np.random.choice(self._n_states, size=1, p=probs)[0])
    
                p_reward = self.R_model[p_state, p_action, p_next_state]
    
                # Updating the Q Function once again using the information of the env model
                self.update_q(reward=p_reward,
                              state=p_state,
                              action=p_action,
                              next_state=p_next_state,
                              done=False)
    
            next_action = greedy_action(next_state, self.Q) if done else epsilon_greedy(next_state, self.Q, self.epsilon)
    
            self.update_time_step()
    
            return (next_action, self.value_function())