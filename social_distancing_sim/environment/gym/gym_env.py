import copy
import os
from typing import Tuple, Any, Dict, Union, List

import gym
import numpy as np

from social_distancing_sim.environment import Environment
from social_distancing_sim.templates.template_base import TemplateBase


class GymEnv(gym.Env):
    """Create a OpenAI Gym environments to handle an social-distancing-sim template environments."""
    template: TemplateBase
    sds_env: Environment

    def __init__(self, env: Union[Environment, None] = None, save_dir: str = ''):
        """
        Wrap an SDS env with a Gym interface.

        The internal SDS env is accessible in .sds_env. The internal observation space and other info are returned in
        the info object on step.

        :param env: Either an Environment object to wrap, or None. If None, expects to build the env from .template.
                    .template is set in the child classes for the registered envs in .gym.environments.
        :param save_dir: Sub dir to save Environment output to, if any.
        """
        self.save_dir = save_dir
        self.save_path: str
        self._set_internal_env(env)
        self._set_observation_space()
        self._set_action_space()

    def _set_internal_env(self, env: Union[Environment, None] = None) -> None:
        """Can either build using custom supplied env, or from set in child template (eg., registered envs)."""
        if env is None:
            env = self.template.build()
        self.sds_env: Environment = env

        # Set the new save paths
        self.save_path = os.path.join(self.sds_env.name, self.save_dir)
        self.sds_env.environment_plotting.set_output_path(self.save_path)

    def _set_observation_space(self) -> None:
        total_pop = self.sds_env.observation_space.graph.total_population
        self.observation_space = gym.spaces.tuple.Tuple((gym.spaces.box.Box(low=0, high=total_pop, shape=(6,),
                                                                            dtype=np.int16),
                                                         gym.spaces.box.Box(low=0, high=1, shape=(total_pop, total_pop),
                                                                            dtype=np.int8),
                                                         gym.spaces.box.Box(low=0, high=1, shape=(total_pop, 5),
                                                                            dtype=np.int8)))

    def _set_action_space(self) -> None:
        self.action_space = gym.spaces.discrete.Discrete(n=5)

    def render(self, mode: str = 'human', show: bool = False, save: bool = True) -> None:
        self.sds_env.plot(plot=show, save=save)

    def step(self, actions_targets: Union[int,
                                          Tuple[List[int], List[Union[int, None]]]],
             ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                        float, bool, Dict[Any, Any]]:

        """
        Step with actions and targets.

        Needs to take a single input.

        :param actions_targets: Either a tuple containing ([actions], [targets]) OR a single int. The single int comes
                                from training an agent with agent.train (and the rlk implementations of .play_episode).
                                In this case, the agent is learning with a single action per turn, which is
                                probably not optimal...
        """

        # Return the sds internal observation space in info for convenience. self.state returns a more limited set of
        # arrays for state, which are derived from the same internal state.

        if isinstance(actions_targets, (int, np.integer)):
            actions_targets = ([actions_targets], [])
        actions, targets = actions_targets

        info, reward, done = self.sds_env.step(actions=actions,
                                               targets=targets)
        return self.state, reward, done, info

    @property
    def state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.sds_env.observation_space.state

    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.sds_env = self.sds_env.clone()
        return self.state

    def replay(self):
        self.sds_env.replay()

    def clone(self) -> "GymEnv":
        return copy.deepcopy(self)
