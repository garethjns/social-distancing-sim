import warnings
from dataclasses import dataclass
from typing import Iterable, Union

from tqdm import tqdm

from social_distancing_sim.agent.agent_base import AgentBase
from social_distancing_sim.agent.dummy_agent import DummyAgent
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
                actions = self.agent.select_actions(obs=self.env.observation_space)

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


if __name__ == "__main__":
    from social_distancing_sim.environment.graph import Graph
    from social_distancing_sim.environment.healthcare import Healthcare
    from social_distancing_sim.environment.observation_space import ObservationSpace
    from social_distancing_sim.environment.environment import Environment
    from social_distancing_sim.disease.disease import Disease
    from social_distancing_sim.environment.environment_plotting import EnvironmentPlotting
    import social_distancing_sim.agent as agents

    seed = 100

    env = Environment(name="example environment",
                      disease=Disease(name='COVID-19',
                                      virulence=0.1,
                                      duration_mean=5,
                                      seed=seed,
                                      immunity_mean=0.95,
                                      immunity_decay_mean=0.05),
                      healthcare=Healthcare(capacity=5),
                      observation_space=ObservationSpace(graph=Graph(community_n=15,
                                                                     community_size_mean=10,
                                                                     community_p_in=1,
                                                                     community_p_out=0.3,
                                                                     seed=seed + 1),
                                                         test_rate=1,
                                                         seed=seed + 2),
                      environment_plotting=EnvironmentPlotting(ts_fields_g2=["Turn score", "Action cost",
                                                                             "Overall score"],
                                                               ts_obs_fields_g2=["Observed turn score", "Action cost",
                                                                                 "Observed overall score"]),
                      seed=seed + 3)

    sim = Sim(env=env,
              n_steps=150,
              agent=agents.VaccinationAgent(seed=8,
                                            actions_per_turn=10),
              tqdm_on=True,
              plot=True,
              save=False)

    sim.run()
