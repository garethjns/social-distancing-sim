import abc
import copy
from typing import Union, Tuple, List, Any

import numpy as np
from reinforcement_learning_keras.agents.agent_base import AgentBase


class LearningAgentBase(AgentBase):
    """Adds compatibility with SDS interface (action selection etc)."""

    @abc.abstractmethod
    def get_best_action(self, state: Any):
        """This method is set by specific RLK agents."""
        pass

    def get_actions(self, state: Union[np.ndarray, Tuple[np.ndarray, np.ndarray],
                                       Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                    training: bool = False) -> Union[Tuple[List[int], List[int]],
                                                     Tuple[List[int], None]]:
        """
        Get next set of actions and targets and track.

        TODO: Handle multiple actions (5 hardcoded below)
        """
        if self.env is None:
            raise AttributeError(f"Not env set, set with agent.set_env()")

        actions = [self.get_best_action(state)] * 5

        return actions, []

    def clone(self) -> "AgentBase":
        """Clone a fresh object with same seed (could be None)"""
        self.unready()
        clone = copy.deepcopy(self)
        self.check_ready()

        return clone

    def set_env(self, *args, **kwargs):
        self.check_ready()
