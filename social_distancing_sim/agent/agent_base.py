import abc
import copy
from typing import List, Union, Dict

import numpy as np

from social_distancing_sim.environment.action_space import ActionSpace
from social_distancing_sim.environment.observation_space import ObservationSpace


class AgentBase(metaclass=abc.ABCMeta):
    action_space = ActionSpace()

    def __init__(self,
                 name: str = 'unnamed_agent',
                 seed: Union[None, int] = None,
                 actions_per_turn: int = 5,
                 start_step: Union[Dict[str, int], None] = None,
                 end_step: Union[Dict[str, int], None] = None) -> None:
        """

        :param name: Agent name.
        :param seed: Seed.
        :param actions_per_turn: Number of actions to return each turn. Automatically limited by available targets each
                                 turn, if necessary.
        :param start_step: Dict keyed by action names, with ints indicating step to start performing actions.
        :param end_step: Dict keyed by action names, with ints indicating step to stop performing actions.
        """
        self.seed = seed
        self.name = name

        if start_step is None:
            start_step = {}
        self.start_step = start_step
        if end_step is None:
            end_step = {}
        self.end_step = end_step
        self.actions_per_turn = actions_per_turn
        self._step = 0  # Track steps as number of .sample calls to agent
        self._prepare_random_state()

    def _prepare_random_state(self) -> None:
        self._random_state = np.random.RandomState(seed=self.seed)

    @property
    def available_actions(self) -> List[str]:
        """
        By default Return all available actions in action space.

        Overload to filter for specific agents.
        """
        return self.action_space.available_actions

    @property
    def currently_active_actions(self) -> List[str]:
        """Return the current active actions based on defined time periods."""

        active_actions = [a for a in self.available_actions
                          if (self._step >= self.start_step.get(a, 0)) and (self._step <= self.end_step.get(a, np.inf))]

        return active_actions

    @staticmethod
    def available_targets(obs: ObservationSpace) -> List[int]:
        """
        By default Return all alive from observation space as potential targets.

        Overload to filter for specific agents.
        """
        return obs.current_alive_nodes

    def _check_available_targets(self, obs: ObservationSpace) -> int:
        """
        Check there are enough available targets to perform requested actions, if not limit n actions.

        :param obs: Observation space to get targets from.
        :return: Number of possible actions given available targets.
        """

        n_available_targets = len(self.available_targets(obs))
        return min(self.actions_per_turn, n_available_targets)

    @abc.abstractmethod
    def select_actions(self, obs: ObservationSpace) -> Dict[int, str]:
        """
        Overload this method to apply agent specific logic for setting actions and targets.

        Can call ._check_available_targets if needed.
        """
        pass

    def get_actions(self, obs: ObservationSpace) -> Dict[int, str]:
        """Get next set of actions and targets and track."""

        actions = self.select_actions(obs)
        self._step += 1
        return actions

    def sample(self, obs: ObservationSpace,
               track: bool = True) -> Dict[int, str]:
        """
        Randomly return self.actions_per_turn actions and targets and optionally track.

        :param obs: ObservationSpace.
        :param track: If True, track in self._steps. Can be set to False if desired.
        """
        n = self._check_available_targets(obs)

        # Randomly pick n actions and targets
        actions = self._random_state.choice(self.available_actions,
                                            size=n)
        targets = self._random_state.choice(self.available_targets(obs),
                                            replace=False,
                                            size=n)

        if track:
            self._step += 1

        return {t: a for t, a in zip(targets, actions)}

    def clone(self) -> "AgentBase":
        """Clone a fresh object with same seed (could be None)."""
        clone = copy.deepcopy(self)
        clone._prepare_random_state()
        return clone

    def reset(self):
        self._step = 0
        self._prepare_random_state()
