from dataclasses import dataclass
from typing import Callable, Any

import numpy as np


@dataclass
class Epsilon:
    initial: float = 0.05
    decay: float = 0.0001

    def __post_init__(self):
        self.eps = self.initial

    def select(self, greedy_option: Callable, random_option: Callable,
               training: bool = False) -> Any:
        """
        If training, apply epsilon greedy and decay epsilon. If not, just return best action.

        Use of lambdas is to avoid unnecessary prediction call if random action is chosen.

        :param s: State to use to get action.
        :param training: Bool indicating if call is during training and to use epsilon greedy and decay.
        :return: Selected action id.
        """
        if training:
            self.eps = self.eps - self.eps * self.decay
            if np.random.random() < self.eps:
                return random_option()

        return greedy_option()
