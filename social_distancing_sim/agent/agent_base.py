import abc
import copy
from typing import List, Union, Dict, Tuple

import numpy as np

from social_distancing_sim.environment.environment import Environment


class AgentBase(metaclass=abc.ABCMeta):

    def __init__(self, env: Union[Environment, None] = None,
                 name: str = 'unnamed_agent',
                 seed: Union[None, int] = None,
                 actions_per_turn: int = 5,
                 start_step: Union[Dict[str, int], None] = None,
                 end_step: Union[Dict[str, int], None] = None) -> None:
        """

        :param env: Reference to environment caller/simulator will be iterating on. Doesn't need to be provided on init,
                    but if not needs to be set with .set_env() before agent will work.
                    Is voided if the agent is cloned, and will need to be reset.
        :param name: Agent name.
        :param seed: Seed.
        :param actions_per_turn: Number of actions to return each turn. Automatically limited by available targets each
                                 turn, if necessary.
        :param start_step: Dict keyed by action names, with ints indicating step to start performing actions.
        :param end_step: Dict keyed by action names, with ints indicating step to stop performing actions.
        """
        self.seed = seed
        self.name = name
        self.actions_per_turn = actions_per_turn

        if start_step is None:
            start_step = {}
        self.start_step = start_step
        self._start_step = None

        if end_step is None:
            end_step = {}
        self.end_step = end_step
        self._end_step = None

        self._step = 0  # Track steps as number of .sample calls to agent
        self._prepare_random_state()

        self.set_env(env)

    def _prepare_random_state(self) -> None:
        self._random_state = np.random.RandomState(seed=self.seed)

    def set_env(self, env: Union[None, Environment]):
        """
        Attach (or detach) the agent to an environment reference.

        This environment is used to access the observations space so the agent can get things like the current infected
        nodes while outside of the environment. It should be a reference, not a copy, or the agent will act of incorrect
        information. Cloning the agent specifically detaches the agent from the environment to avoid mistakes.
        Where something holding both the environment and agent is cloned (for example, Sims when running MultiSims)
        the agent's reference is reset to None and the correct environment needs to be reattached with
        agent.set_env(new_ref).
        """

        self.env = env

        if self.env is not None:
            if self._start_step is None:
                self._start_step = {self.env.action_space.get_action_id(a): v for a, v in self.start_step.items()}
            if self._end_step is None:
                self._end_step = {self.env.action_space.get_action_id(a): v for a, v in self.end_step.items()}
        else:
            self._start_step = None
            self._end_step = None

    @property
    def available_actions(self) -> List[int]:
        """
        By default Return all available actions in action space.

        Overload to filter for specific agents.
        """
        return self.env.action_space.available_action_ids

    @property
    def currently_active_actions(self) -> List[int]:
        """Return the current active actions based on defined time periods."""

        active_actions = [a for a in self.available_actions
                          if (self._step >= self._start_step.get(a, 0))
                          and (self._step <= self._end_step.get(a, np.inf))]

        return active_actions

    @property
    def available_targets(self) -> List[int]:
        """
        By default Return all alive from observation space as potential targets.

        Overload to filter for specific agents.
        """
        return self.env.observation_space.current_alive_nodes

    def _check_available_targets(self) -> int:
        """
        Check there are enough available targets to perform requested actions, if not limit n actions.

        :return: Number of possible actions given available targets.
        """
        return min(self.actions_per_turn, len(self.available_targets))

    @abc.abstractmethod
    def _select_actions_targets(self) -> Dict[int, int]:
        """
        Overload this method to apply agent specific logic for setting actions and targets.

        Can call ._check_available_targets if needed.
        """
        pass

    def get_actions(self, state: Union[np.ndarray, Tuple[np.ndarray, np.ndarray],
                                       Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                    training: bool = False) -> Union[Tuple[List[int], List[int]],
                                                     Tuple[List[int], None]]:
        """Get next set of actions and targets and track."""
        if self.env is None:
            raise AttributeError(f"Not env set, set with agent.set_env()")

        actions_dict = self._select_actions_targets()
        self._step += 1
        return list(actions_dict.values()), list(actions_dict.keys())

    def sample(self,
               track: bool = True) -> Dict[int, int]:
        """
        Randomly return self.actions_per_turn actions and targets and optionally track.

        :param track: If True, track in self._steps. Can be set to False if desired.
        """
        n = self._check_available_targets()

        # Randomly pick n actions and targets
        actions = self._random_state.choice(self.available_actions,
                                            size=n)
        targets = self._random_state.choice(self.available_targets,
                                            replace=False,
                                            size=n)

        if track:
            self._step += 1

        return {t: a for t, a in zip(targets, actions)}

    def clone(self) -> "AgentBase":
        """Clone a fresh object with same seed (could be None)."""
        self.set_env(None)
        clone = copy.deepcopy(self)
        clone._prepare_random_state()
        return clone

    def reset(self):
        self._step = 0
        self._prepare_random_state()

    def update(self, *args, **kwargs):
        """To match RL agent interface, skip any update call."""
        pass
