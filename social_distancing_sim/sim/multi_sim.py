import multiprocessing
from dataclasses import dataclass
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from social_distancing_sim.agent.vaccination_agent import VaccinationAgent
from social_distancing_sim.population.history import History
from social_distancing_sim.sim.sim import Sim


@dataclass
class MultiSim:
    sim: Sim
    n_reps: int = 100
    n_jobs: int = multiprocessing.cpu_count() - 2
    name: str = 'Unnamed experiment'

    def __post_init__(self):
        self._mlflow_exp = None
        self.results = pd.DataFrame()

    def _run(self):
        sim = self.sim.clone()
        results = sim.run()
        return results

    def run(self):
        results = Parallel(n_jobs=self.n_jobs,
                           backend='loky')(delayed(self._run)() for _ in tqdm(range(self.n_reps),
                                                                              desc=self.sim.pop.name))

        # Place in fake history container for now
        results_hist = History()
        for h in results:
            results_hist.log({k: v[0] for k, v in h.items()})

        self.results = pd.DataFrame(results_hist)
        self.log()

    def _agg_stats(self, x: pd.Series) -> Dict[str, float]:
        m = x.mean()
        ci = np.percentile(x, [2.5, 97.5])
        name = "".join([s for s in str(x.name) if s.isalpha() or (s in ['_', '.', '-', ' ', '/'])])
        return {f"{name}__mean": m,
                f"{name}__ci_lb": ci[0],
                f"{name}__ci_ub": ci[1]}

    def log(self):
        self._mlflow_exp = mlflow.set_experiment(self.name)
        mlflow.start_run()
        mlflow.log_params({'sim_n_steps': self.sim.n_steps,
                           'pop_total_population': self.sim.pop.total_population,
                           'pop_name': self.sim.pop.name,
                           'pop_random_infection_chance': self.sim.pop.random_infection_chance,
                           'disease_name': self.sim.pop.disease.name,
                           'disease_virulence': self.sim.pop.disease.virulence,
                           'recovery_rate': self.sim.pop.disease.recovery_rate,
                           'duration_mean': self.sim.pop.disease.duration_mean,
                           'duration_std': self.sim.pop.disease.duration_std,
                           'immunity_mean': self.sim.pop.disease.immunity_mean,
                           'immunity_std': self.sim.pop.disease.immunity_std,
                           'immunity_decay_mean': self.sim.pop.disease.immunity_decay_mean,
                           'immunity_decay_std': self.sim.pop.disease.immunity_decay_std,
                           'obs_test_rate': self.sim.pop.observation_space.test_rate,
                           'obs_test_validity_period': self.sim.pop.observation_space.test_validity_period,
                           'graph_community_n': self.sim.pop.observation_space.graph.community_n,
                           'graph_community_size_mean': self.sim.pop.observation_space.graph.community_size_mean,
                           'graph_community_size_std': self.sim.pop.observation_space.graph.community_size_std,
                           'graph_community_p_in': self.sim.pop.observation_space.graph.community_p_in,
                           'graph_community_p_out': self.sim.pop.observation_space.graph.community_p_out,
                           'graph_considered_immune_threshold':
                               self.sim.pop.observation_space.graph.considered_immune_threshold,
                           'scoring_clear_yield_per_edge': self.sim.pop.scoring.clear_yield_per_edge,
                           'scoring_infection_penalty': self.sim.pop.scoring.infection_penalty,
                           'scoring_death_penalty': self.sim.pop.scoring.death_penalty,
                           'agent_name': self.sim.agent.name,
                           'agent_type': self.sim.agent.__class__.__name__,
                           'agent_delay': self.sim.agent_delay,
                           'agent_actions_per_turn': self.sim.agent.actions_per_turn,
                           'agent_action_space_vaccinate_cost': self.sim.agent.action_space.vaccinate_cost,
                           'agent_action_space_isolate_cost': self.sim.agent.action_space.isolate_cost})

        metrics_to_log = {}
        for c in ["Observed overall score", "Observed score", "Overall score", "Score"]:
            metrics_to_log.update(self._agg_stats(self.results[c]))
        mlflow.log_metrics(metrics_to_log)

        mlflow.end_run()

if __name__ == "__main__":
    from social_distancing_sim.population.graph import Graph
    from social_distancing_sim.population.healthcare import Healthcare
    from social_distancing_sim.population.observation_space import ObservationSpace
    from social_distancing_sim.population.population import Population
    from social_distancing_sim.disease.disease import Disease

    seed = None

    pop = Population(name="multi sim population",
                     disease=Disease(name='COVID-19',
                                     virulence=0.01,
                                     seed=seed,
                                     immunity_mean=0.95,
                                     immunity_decay_mean=0.05),
                     healthcare=Healthcare(capacity=5),
                     observation_space=ObservationSpace(graph=Graph(community_n=15,
                                                                    community_size_mean=10,
                                                                    seed=seed),
                                                        test_rate=1,
                                                        seed=seed),
                     seed=seed,
                     plot_ts_fields_g2=["Score", "Action cost", "Overall score"],
                     plot_ts_obs_fields_g2=["Observed Score", "Action cost", "Observed overall score"])

    sim = Sim(pop=pop,
              n_steps=150,
              agent_delay=30,
              agent=VaccinationAgent(actions_per_turn=10,
                                     seed=seed))

    multi_sim = MultiSim(sim)
    multi_sim.run()
