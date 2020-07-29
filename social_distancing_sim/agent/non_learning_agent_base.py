import abc
import copy
from typing import List, Union, Dict, Tuple

import gym
import numpy as np
from gym.envs.registration import EnvSpec
from reinforcement_learning_keras.agents.components.helpers.env_builder import EnvBuilder

from social_distancing_sim.environment.action_space import ActionSpace
from social_distancing_sim.environment.gym.gym_env import GymEnv


class NonLearningAgentBase(metaclass=abc.ABCMeta):
    """
    Base for non-rl agents.

    NonLearningAgents always used gym interface to environment, but are free to bypass this and access env.sds_env
    """

    def __init__(self, env_spec: Union[str, None] = None,
                 name: str = 'unnamed_agent',
                 seed: Union[None, int] = None,
                 actions_per_turn: int = 5,
                 start_step: Union[Dict[str, int], None] = None,
                 end_step: Union[Dict[str, int], None] = None) -> None:
        """

        :param env_spec: Name of registered env. Optional. If agent is not attached to an env, it will build one from
                         this spec. If spec is not provided, agent should be attached to an existing env with
                         self.attach_to_env(env).
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

        self._prepare_random_state()

        # If env not attached with spec, should be done manually
        self.env = env_spec
        self._set_action_ranges(start_step, end_step)

    @property
    def env(self) -> GymEnv:
        return self._env

    @env.setter
    def env(self, env_or_spec: Union[str, GymEnv]) -> GymEnv:
        if isinstance(env_or_spec, str):
            # Using own env spec to build env as needed
            self.env_builder = EnvBuilder(env_or_spec)
            self._env = self.env_builder.env
        else:
            # Either using specified env that may also contain other agents, or intending to. If latter, env_or_spec
            # will be None at this point.
            self._env = env_or_spec

    def attach_to_env(self, env_or_spec: Union[GymEnv, str, gym.envs.registration.EnvSpec]) -> None:
        """
        Attach agent to an existing env.

        Used by MultiAgent to attach children agents to the same environment (rather than children building their own
        from spec)
        """
        if isinstance(env_or_spec, str):
            self.env = gym.make(env_or_spec)
        elif isinstance(env_or_spec, EnvSpec):
            self.env = env_or_spec.make()
        else:
            # Assuming instantiated env like GymEnv, or wrapped version, eg. TimeLimit, etc.
            self.env = env_or_spec

    def _prepare_random_state(self) -> None:
        self._random_state = np.random.RandomState(seed=self.seed)

    def _set_action_ranges(self, start_step: Union[Dict[str, int], None],
                           end_step: Union[Dict[str, int], None]) -> None:
        if start_step is None:
            start_step = {}
        self.start_step = start_step
        self._start_step = None

        if end_step is None:
            end_step = {}
        self.end_step = end_step
        self._end_step = None

        self._step = 0  # Track steps as number of .sample calls to agent

        if self._start_step is None:
            self._start_step = {ActionSpace().get_action_id(a): v
                                for a, v in self.start_step.items()}
        if self._end_step is None:
            self._end_step = {ActionSpace().get_action_id(a): v
                              for a, v in self.end_step.items()}

    @property
    def available_actions(self) -> List[int]:
        """
        By default Return all available actions in action space.

        Overload to filter for specific agents.
        """
        return self.env.sds_env.action_space.available_action_ids

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
        return self.env.sds_env.observation_space.current_alive_nodes

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
            raise AttributeError(f"No env set, set with agent.attach_to_env()")

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

    def clone(self) -> "NonLearningAgentBase":
        """Clone a fresh object with same seed (could be None)."""
        self.env = None
        clone = copy.deepcopy(self)
        clone._prepare_random_state()
        return clone

    def reset(self):
        self._step = 0
        self._prepare_random_state()
