import abc
import copy
from typing import List, Union, Dict, Tuple

import numpy as np

from social_distancing_sim.environment.action_space import ActionSpace
from social_distancing_sim.environment.observation_space import ObservationSpace


class AgentBase(metaclass=abc.ABCMeta):
    action_space = ActionSpace()

    def __init__(self,
                 name: str = 'unnamed_agent',
                 seed: Union[None, int] = None,
                 actions_per_turn: int = 5) -> None:
        self.seed = seed
        self.name = name
        self.actions_per_turn = actions_per_turn
        self._prepare_random_state()

    def _prepare_random_state(self) -> None:
        self.state = np.random.RandomState(seed=self.seed)

    @property
    def available_actions(self):
        """
        By default Return all available actions in action space.

        Overload to filter for specific agents.
        """
        return self.action_space.available_actions

    @staticmethod
    def available_targets(obs: ObservationSpace):
        """
        By default Return all alive from observation space as potential targets.

        Overload to filter for specific agents.
        """
        return obs.current_alive_nodes

    def _check_available_targets(self, obs: ObservationSpace, n: int) -> int:
        """
        Check there are enough available targets to perform requested actions, if not limit n actions.

        :param obs: Observation space to get targets from.
        :param n: Number of requested actions.
        :return: Number of possible actions given available targets.
        """
        n_available_targets = len(self.available_targets(obs))
        return min(n, n_available_targets)

    @abc.abstractmethod
    def select_actions(self, obs: ObservationSpace,
                       n: int = 1) -> Dict[int, str]:
        """
        Overload this method to apply agent specific logic for setting actions and targets.

        Can call ._check_available_targets if needed.
        """
        pass

    def sample(self, obs: ObservationSpace, n: int) -> Dict[int, str]:
        n = self._check_available_targets(obs, n)

        # Randomly pick n actions and targets
        actions = self.state.choice(self.available_actions,
                                    size=n)
        targets = self.state.choice(self.available_targets(obs),
                                    replace=False,
                                    size=n)

        return {t: a for t, a in zip(targets, actions)}

    def clone(self) -> "AgentBase":
        """Clone a fresh object with same seed (could be None)."""
        clone = copy.deepcopy(self)
        clone._prepare_random_state()
        return clone
