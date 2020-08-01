import os
import shutil
import warnings
from dataclasses import dataclass
from typing import Iterable, Union, Any

import gym
from tqdm import tqdm

from social_distancing_sim.agent import DummyAgent
from social_distancing_sim.agent.learning_agent_base import LearningAgentBase
from social_distancing_sim.agent.non_learning_agent_base import NonLearningAgentBase
from social_distancing_sim.environment.gym.gym_env import GymEnv
from social_distancing_sim.environment.history import History


@dataclass
class Sim:
    """
    Agent evaluation class (no training).
    """
    env_spec: gym.envs.registration.EnvSpec
    save_dir: str = 'sim'
    agent: Union[NonLearningAgentBase, LearningAgentBase] = DummyAgent()
    training: bool = False
    n_steps: int = 100
    plot: bool = False
    save: bool = False
    tqdm_on: bool = False
    logging: bool = False

    def __post_init__(self):
        self.env: GymEnv

        if self.tqdm_on:
            self._tqdm = tqdm

        self._step: int = 0

    def _prepare_agent(self) -> Any:
        """
        Prepare the agent; if there isn't one create a Dummy.

        Set the env for the agent, which will be env that's stepped in the Sim. This. This should handle cases where
        some agents are using wrappers and others aren't.

        This agent is reset here, as reset rebuilds from spec, resetting logging options (which default to off).
        These are set after reset.
        """
        self.env = self.env_spec.make()
        self.agent.attach_to_env(self.env)
        initial_obs = self.agent.env.reset()

        # Set the new save paths
        self.save_path = os.path.join(self.save_dir, self.env.save_path, self.agent.name)
        shutil.rmtree(self.save_path, ignore_errors=True)
        self.agent.env.sds_env.set_output_path(self.save_path)
        self.agent.env.sds_env.environment_plotting.set_output_path(self.save_path)
        self.agent.env.sds_env.log_to_file = self.logging

        return initial_obs

    @staticmethod
    def _tqdm(x: Iterable, *args, **kwargs) -> Iterable:
        return x

    def step(self) -> None:
        self.agent.env.sds_env.logger.info(f"\n\n***Sim step: {self._step}***")

        # Pick action
        actions, targets = self.agent.get_actions(state=self._last_state)
        self.agent.env.sds_env.logger.info(f"Agent requested actions {actions} with targets {targets}")

        # Step the simulation and observe for this step
        observation, reward, done, info = self.agent.env.step((actions, targets))
        self.agent.env.sds_env.logger.info(f"Environment returned reward {reward}")
        self.agent.env.sds_env.logger.info(f"Environment done={done}")
        self._last_state = observation

        self.agent.env.sds_env.plot(plot=self.plot, save=self.save)

    def run(self) -> History:
        self._last_state = self._prepare_agent()
        self.agent.env.sds_env._total_steps = self.n_steps
        self.agent.env.sds_env.plot(plot=self.plot, save=self.save)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=UserWarning)

            for _ in self._tqdm(range(self.n_steps),
                                desc=f"{self.agent.env.sds_env.name}: {self.agent.name}"):
                self.step()
                self._step += 1

        if self.save:
            self.agent.env.sds_env.replay()

        return self.history

    @property
    def history(self) -> History:
        return self.agent.env.sds_env.history

    def clone(self) -> "Sim":
        """Clone a fresh object with same seed (could be None)."""
        return Sim(env_spec=self.env_spec,
                   agent=self.agent.clone(),
                   n_steps=self.n_steps, plot=self.plot,
                   save=self.save, tqdm_on=self.tqdm_on)
