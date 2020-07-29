import abc
import copy
from functools import reduce
from typing import Union, Tuple, List, Type, Any

import gym
import numpy as np
from gym.envs.registration import EnvSpec
from reinforcement_learning_keras.agents.agent_base import AgentBase
from reinforcement_learning_keras.agents.components.helpers.env_builder import EnvBuilder

from social_distancing_sim.environment.gym.gym_env import GymEnv


class LearningAgentBase(metaclass=abc.ABCMeta):
    """
    Defines the required interface to add compatibility between RLK agent interface and SDS interface

    - Multiple action selection
    - Targeted actions
    - Cloning
    - Env attachment
    """

    rlk_agent_class: Type[AgentBase]

    def __init__(self, *args, actions_per_turn: int = 5, **kwargs):
        self.actions_per_turn = actions_per_turn
        if len(kwargs.keys()) > 0:
            self.rlk_agent = self.rlk_agent_class(*args, **kwargs)

    @abc.abstractmethod
    def get_actions(self,
                    state: Union[np.ndarray, Tuple[np.ndarray, np.ndarray],
                                 Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
                    ) -> Union[Tuple[List[int], List[int]], Tuple[List[int], None]]:
        """
        Get the actions from the agent.

        Used by Sim, SDS interface. Should return same as get_action.
        """
        pass

    def get_action(self, *args, **kwargs):
        """Used by rlk_agent.train."""
        return self.get_actions(*args, **kwargs)

    def clone(self) -> "LearningAgentBase":
        """Clone a fresh object with same seed (could be None)"""
        self.rlk_agent.unready()
        clone = copy.deepcopy(self)
        self.rlk_agent.check_ready()

        return clone

    def attach_to_env(self, env_or_spec: Union[GymEnv, str, gym.envs.registration.EnvSpec]) -> None:
        """
        This might be a bit inconsistent between NonLearningAgent and LearningAgent.

        Needs to handle wrappers, which is what EnvBuilder is designed to do. However env builder only supports
        .making-ing from a str, not from a EnvSpec or existing env. This needs to be handled here, until EnvBuilder is
        upgraded.

        TODO: This is brittle and needs to be simplified.
        """

        # This ensures an env_builder already exists. It'll either be modified or completely replaced below.
        self.rlk_agent.check_ready()

        if isinstance(env_or_spec, str):
            # Supported by EnvBuilder
            # If str, build with env builder. Reuse current wrappers.
            self.rlk_agent.env_builder = EnvBuilder(env_or_spec, self.rlk_agent.env_wrappers)
        else:
            # Not supported by EnvBuilder yet
            # Will need to handle wrapping
            if isinstance(env_or_spec, EnvSpec):
                # If it's a spec, make
                env = env_or_spec.make()
            else:
                # Otherwise, Assuming instantiated env like GymEnv, as above, set for env_builder.
                env = env_or_spec

            # Replace env_builders wrapped version instantiated env. There's no setter for .env yet so just
            # using private attr directly.
            self.env_builder._env = reduce(lambda inner_env, wrapper: wrapper(inner_env),
                                           self.rlk_agent.env_wrappers, env)

    def train(self, *args, **kwargs) -> None:
        self.rlk_agent.train(*args, **kwargs)

    @property
    def env(self) -> Any:
        return self.rlk_agent.env

    @property
    def name(self) -> str:
        return self.rlk_agent.name

    @property
    def env_builder(self) -> EnvBuilder:
        return self.rlk_agent.env_builder

    def play_episode(self, *args, **kwargs):
        self.rlk_agent.play_episode(*args, **kwargs)

    @classmethod
    def load(cls, fn: str, actions_per_turn: int = 5) -> "LearningAgentBase":
        agent = cls(actions_per_turn=actions_per_turn)
        agent.rlk_agent = cls.rlk_agent_class.load(fn)

        return agent

    def check_ready(self):
        self.rlk_agent.check_ready()

    def save(self) -> None:
        self.rlk_agent.save()
