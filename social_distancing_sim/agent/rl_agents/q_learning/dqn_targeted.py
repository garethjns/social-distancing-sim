import numpy as np
from reinforcement_learning_keras.agents.q_learning.deep_q_agent import DeepQAgent as RLKDeepQAgent


class DeepQAgent(RLKDeepQAgent):
    def get_action(self, s: np.ndarray, training: bool = False) -> int:
        pass

    def get_best_action(self, s: np.ndarray) -> np.ndarray:
        pass
