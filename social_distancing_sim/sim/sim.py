import warnings
from dataclasses import dataclass
from typing import Dict, Any, Iterable

from tqdm import tqdm

from social_distancing_sim.agent.agent import Agent
from social_distancing_sim.agent.vaccination_agent import VaccinationAgent
from social_distancing_sim.population.history import History
from social_distancing_sim.population.population import Population


@dataclass
class Sim:
    pop: Population
    agent: Agent
    n_steps: int = 100
    plot: bool = False
    save: bool = False
    agent_delay: int = 1
    tqdm_on: bool = False

    def __post_init__(self):
        if self.tqdm_on:
            self._tqdm = tqdm

    @staticmethod
    def _tqdm(x: Iterable, *args, **kwargs) -> Iterable:
        return x

    def _log(self, completed_actions: Dict[str, Any], action_cost: float = 0):
        if len(self.pop.history["Score"]) > 0:
            score = self.pop.history["Score"][-1]
            obs_score = self.pop.history["Observed score"][-1]
        else:
            score = 0
            obs_score = 0

        self.pop.history.log({'Action cost': action_cost,
                              'Overall score': score - action_cost,
                              'Observed overall score': obs_score - action_cost,
                              'Completed actions': completed_actions})

    def run(self) -> History:
        self.pop._total_steps = self.n_steps

        # TODO: Might want to add own history and plotting rather than using populations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=(UserWarning, RuntimeWarning))
            for s in self._tqdm(range(self.n_steps),
                                desc=self.pop.name):
                # Step the simulation
                self.pop.step()

                # Act
                completed_actions = {}
                action_cost = 0
                if s > self.agent_delay:
                    completed_actions, action_cost = self.agent.act(self.pop,
                                                                    step=s)

                # Log action related info to population object
                self._log(completed_actions=completed_actions,
                          action_cost=action_cost)

                # PLot population after logging so sim-added logs are available to population history
                if self.plot or self.save:
                    self.pop.plot(show=self.plot,
                                  save=self.save)

        final_hist = History()
        for k, v in self.pop.history.items():
            # Skip any non-sensible ones
            if k in ["Completed actions"]:
                continue
            final_hist.log({k: v[-1]})

        return final_hist

    def clone(self) -> "Sim":
        """Clone a fresh object with same seed (could be None)."""
        return Sim(pop=self.pop.clone(), agent=self.agent.clone(), n_steps=self.n_steps, plot=self.plot,
                   save=self.save, tqdm_on=self.tqdm_on, agent_delay=self.agent_delay)


if __name__ == "__main__":
    from social_distancing_sim.population.graph import Graph
    from social_distancing_sim.population.healthcare import Healthcare
    from social_distancing_sim.population.observation_space import ObservationSpace
    from social_distancing_sim.population.population import Population
    from social_distancing_sim.disease.disease import Disease

    seed = 123

    pop = Population(name="example population",
                     disease=Disease(name='COVID-19',
                                     virulence=0.01,
                                     seed=seed,
                                     immunity_mean=0.95,
                                     immunity_decay_mean=0.05),
                     healthcare=Healthcare(capacity=5),
                     observation_space=ObservationSpace(graph=Graph(community_n=15,
                                                                    community_size_mean=10,
                                                                    seed=seed + 1),
                                                        test_rate=1,
                                                        seed=seed + 2),
                     seed=seed + 3,
                     plot_ts_fields_g2=["Score", "Action cost", "Overall score"],
                     plot_ts_obs_fields_g2=["Observed Score", "Action cost", "Observed overall score"])

    sim = Sim(pop=pop,
              agent_delay=10,
              agent=VaccinationAgent(actions_per_turn=25,
                                     seed=seed),
              plot=True,
              save=True)

    sim.run()
