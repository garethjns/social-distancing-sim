import warnings
from dataclasses import dataclass
from typing import Iterable, Union

from tqdm import tqdm

from social_distancing_sim.agent.agent_base import AgentBase
from social_distancing_sim.agent.basic_agents.dummy_agent import DummyAgent
from social_distancing_sim.environment.environment import Environment
from social_distancing_sim.environment.history import History


@dataclass
class Sim:
    env: Environment
    agent: Union[AgentBase, None] = None
    n_steps: int = 100
    plot: bool = False
    save: bool = False
    tqdm_on: bool = False

    def __post_init__(self):
        if self.tqdm_on:
            self._tqdm = tqdm

        if self.agent is None:
            self.agent = DummyAgent()

    @staticmethod
    def _tqdm(x: Iterable, *args, **kwargs) -> Iterable:
        return x

    def run(self) -> History:
        self.env._total_steps = self.n_steps

        # TODO: Might want to add own history and plotting rather than using populations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=(UserWarning, RuntimeWarning))

            for s in self._tqdm(range(self.n_steps),
                                desc=self.env.name):

                # Pick action
                actions = self.agent.get_actions(obs=self.env.observation_space)

                # Step the simulation
                observation, reward, done, info = self.env.step(actions)

                # PLot environment after logging so sim-added logs are available to environment history
                if self.plot or self.save:
                    self.env.environment_plotting.plot(obs=self.env.observation_space,
                                                       history=self.env.history,
                                                       healthcare=self.env.healthcare,
                                                       total_steps=self.n_steps,
                                                       step=s,
                                                       show=self.plot,
                                                       save=self.save)

            final_hist = History()
            for k, v in self.env.history.items():
                # Skip any non-sensible ones
                if k in ["Completed actions"]:
                    continue
                if len(v) > 1:
                    final_hist.log({k: v[-1]})

            return final_hist

    def clone(self) -> "Sim":
        """Clone a fresh object with same seed (could be None)."""
        return Sim(env=self.env.clone(),
                   agent=self.agent.clone() if self.agent is not None else None,
                   n_steps=self.n_steps, plot=self.plot,
                   save=self.save, tqdm_on=self.tqdm_on)
