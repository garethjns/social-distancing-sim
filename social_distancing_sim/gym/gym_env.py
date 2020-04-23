from typing import Tuple, Any, Dict, Union, List

import gym
import numpy as np

from social_distancing_sim.templates.template_base import TemplateBase


class GymEnv(gym.Env):
    """Create a OpenAI Gym environment to handle an social-distancing-sim template environment."""

    def __init__(self, template: TemplateBase,
                 seed: Union[int, None] = None):
        self.seed(seed=seed)
        self._template = template
        self._sds_env = self._template.build()
        total_pop = self._sds_env.observation_space.graph.total_population
        self.action_space = gym.spaces.tuple.Tuple((gym.spaces.box.Box(low=0, high=total_pop, shape=(6,),
                                                                       dtype=np.int8),
                                                    gym.spaces.box.Box(low=0, high=1, shape=(total_pop, total_pop),
                                                                       dtype=np.int8),
                                                    gym.spaces.box.Box(low=0, high=1, shape=(total_pop, 5),
                                                                       dtype=np.int8)))
        self.observation_space = gym.spaces.discrete.Discrete(n=5)

    def render(self, mode: str = 'human'):
        """TODO: Render plot with ._sds_env.environment_plotting."""
        pass

    def step(self, action: Dict[int, str]) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                                                    float, bool, Dict[Any, Any]]:
        _, reward, done = self._sds_env.step(actions=action)

        obs = (self._sds_env.observation_space.state_summary(),
               self._sds_env.observation_space.state_graph(),
               self._sds_env.observation_space.state_nodes())
        info = {}

        return obs, reward, done, info

    def seed(self, seed: Union[int, None] = None) -> List[Union[int, None]]:
        self._environment_seed = 20200423
        self._graph_seed = seed + 10 if seed else None
        self._disease_seed = seed + 1234 if seed else None
        self._observation_space_seed = seed + 231789 if seed else None

        return [self._environment_seed, self._graph_seed, self._disease_seed, self._observation_space_seed]

    def reset(self) -> None:
        self._sds_env = self._template.build()
