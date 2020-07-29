from typing import Union, Tuple, List

import numpy as np
from reinforcement_learning_keras.agents.q_learning.deep_q_agent import DeepQAgent

from social_distancing_sim.agent.learning_agent_base import LearningAgentBase


class DQNUntargeted(LearningAgentBase):
    """Makes RLK DeepQAgent compatible with SDS agent interface."""
    rlk_agent_class = DeepQAgent

    def get_actions(self, state: Union[np.ndarray, Tuple[np.ndarray, np.ndarray],
                                       Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                    training: bool = False) -> Union[Tuple[List[int], List[int]],
                                                     Tuple[List[int], None]]:
        """Get next set of actions and targets and track."""
        if self.rlk_agent.env is None:
            raise AttributeError(f"No env set, set with agent.attach_to_env()")

        actions = [self.rlk_agent.get_best_action(state)] * self.actions_per_turn

        return actions, []
