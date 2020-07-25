import os
import warnings
from dataclasses import dataclass
from typing import Iterable, Union

import gym
from tqdm import tqdm

from social_distancing_sim.agent.learning_agent_base import LearningAgentBase
from social_distancing_sim.agent.non_learning_agent_base import NonLearningAgentBase
from social_distancing_sim.environment.history import History


@dataclass
class Sim:
    """
    Agent evaluation class (no training).
    """
    env_spec: gym.envs.registration.EnvSpec
    save_dir: str = 'sim'
    agent: Union[NonLearningAgentBase, LearningAgentBase, None] = None
    training: bool = False
    n_steps: int = 100
    plot: bool = False
    save: bool = False
    tqdm_on: bool = False

    def __post_init__(self):
        if self.tqdm_on:
            self._tqdm = tqdm

        self._prepare_env()
        self._prepare_agent()

        self._step: int = 0

    def _prepare_env(self):
        """Prepare the env."""
        self.env = self.env_spec.make()

        # Set the new save paths
        self.save_path = os.path.join(self.save_dir, self.env.save_path, self.agent.name)
        self.env.sds_env.environment_plotting.set_output_path(self.save_path)

    def _prepare_agent(self):
        """
        Prepare the agent; if there isn't one create a Dummy.

        Set the env for the agent, which will be env that's stepped in the Sim. This. This should handle cases where
        some agents are using wrappers and others aren't.
        """
        self.agent.attach_to_env(self.env)

    @staticmethod
    def _tqdm(x: Iterable, *args, **kwargs) -> Iterable:
        return x

    def step(self):
        # Pick action
        actions, targets = self.agent.get_actions(state=self._last_state)

        # Step the simulation and observe for this step
        observation, reward, done, info = self.agent.env.step(actions)
        self._last_state = observation

        self.agent.env.sds_env.plot(plot=self.plot, save=self.save)

    def run(self) -> History:
        self.agent.env.sds_env._total_steps = self.n_steps
        self.agent.env.sds_env.plot(plot=self.plot, save=self.save)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=UserWarning)

            self._last_state = self.agent.env.reset()
            for _ in self._tqdm(range(self.n_steps),
                                desc=f"{self.agent.env.sds_env.name}: {self.agent.name}"):
                self.step()
                self._step += 1

            final_hist = History()
            for k, v in self.agent.env.sds_env.history.items():
                # Skip any non-sensible ones
                if k in ["Completed actions"]:
                    continue
                if len(v) > 1:
                    final_hist.log({k: v[-1]})

            return final_hist

    def clone(self) -> "Sim":
        """Clone a fresh object with same seed (could be None)."""
        return Sim(env_spec=self.env_spec,
                   agent=self.agent.clone(),
                   n_steps=self.n_steps, plot=self.plot,
                   save=self.save, tqdm_on=self.tqdm_on)
