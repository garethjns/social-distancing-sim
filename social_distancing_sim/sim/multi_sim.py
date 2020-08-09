import multiprocessing
from dataclasses import dataclass
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from social_distancing_sim.environment.gym.gym_env import GymEnv
from social_distancing_sim.environment.history import History
from social_distancing_sim.sim.sim import Sim


def _dummy_logger(*args, **kwargs) -> None:
    """Turn of logging for MultiSims."""
    print("what")


@dataclass
class MultiSim:
    sim: Sim
    n_reps: int = 100
    n_jobs: int = multiprocessing.cpu_count() - 2
    name: str = 'Unnamed experiment'

    def __post_init__(self):
        self._mlflow_exp = None
        self.results = pd.DataFrame()

        # Create a reference env that will be used in results logging. It's mainly used to log params.
        self.reference_env: GymEnv = self.sim.env_spec.make()

    def _run(self):
        # Clone to make sure the sims are actually run on different environments
        sim = self.sim.clone()
        # Reattach env to agent
        sim.agent.attach_to_env(sim.env_spec)
        sim.agent.env.sds_env.log_to_file = False
        results = sim.run()

        return results

    def run(self):
        self.full_results = Parallel(n_jobs=self.n_jobs,
                                     backend='loky')(delayed(self._run)()
                                                     for _ in tqdm(range(self.n_reps),
                                                                   desc=self.sim.agent.name))

        # Place in fake history container for now
        results_hist = History()
        for h in self.full_results:
            results_hist.log({k: v[-1] for k, v in h.items()})

        self.results = pd.DataFrame(results_hist)
        self.log()

    @staticmethod
    def _agg_stats(x: pd.Series) -> Dict[str, float]:
        m = x.mean()
        ci = np.percentile(x, [2.5, 97.5])
        name = "".join([s for s in str(x.name) if s.isalpha() or (s in ['_', '.', '-', ' ', '/'])])
        return {f"{name}__mean": m,
                f"{name}__ci_lb": ci[0],
                f"{name}__ci_ub": ci[1]}

    def log(self):
        """
        Log parameters of the env used.

        Using Sim's copy here, as iterated env isn't returned from _run. This logs the params, which are the same (ie.
        what's defined in the template).
        """
        self._mlflow_exp = mlflow.set_experiment(self.name)
        mlflow.start_run()
        mlflow.log_params({'sim_n_steps': self.sim.n_steps,
                           'pop_total_population_EXAMPLE':
                               self.reference_env.sds_env.total_population,  # Can vary if not deterministic!
                           'pop_name': self.reference_env.sds_env.name,
                           'pop_random_infection_chance': self.reference_env.sds_env.random_infection_chance,
                           'disease_name': self.reference_env.sds_env.disease.name,
                           'disease_virulence': self.reference_env.sds_env.disease.virulence,
                           'recovery_rate': self.reference_env.sds_env.disease.recovery_rate,
                           'duration_mean': self.reference_env.sds_env.disease.duration_mean,
                           'duration_std': self.reference_env.sds_env.disease.duration_std,
                           'immunity_mean': self.reference_env.sds_env.disease.immunity_mean,
                           'immunity_std': self.reference_env.sds_env.disease.immunity_std,
                           'immunity_decay_mean': self.reference_env.sds_env.disease.immunity_decay_mean,
                           'immunity_decay_std': self.reference_env.sds_env.disease.immunity_decay_std,
                           'obs_test_rate': self.reference_env.sds_env.observation_space.test_rate,
                           'obs_test_validity_period':
                               self.reference_env.sds_env.observation_space.test_validity_period,
                           'graph_community_n': self.reference_env.sds_env.observation_space.graph.community_n,
                           'graph_community_size_mean':
                               self.reference_env.sds_env.observation_space.graph.community_size_mean,
                           'graph_community_size_std':
                               self.reference_env.sds_env.observation_space.graph.community_size_std,
                           'graph_community_p_in': self.reference_env.sds_env.observation_space.graph.community_p_in,
                           'graph_community_p_out': self.reference_env.sds_env.observation_space.graph.community_p_out,
                           'graph_considered_immune_threshold':
                               self.reference_env.sds_env.observation_space.graph.considered_immune_threshold,
                           'scoring_clear_yield_per_edge': self.reference_env.sds_env.scoring.clear_yield_per_edge,
                           'scoring_infection_penalty': self.reference_env.sds_env.scoring.infection_penalty,
                           'scoring_death_penalty': self.reference_env.sds_env.scoring.death_penalty,
                           'agent_name': self.sim.agent.name,
                           'agent_type': self.sim.agent.__class__.__name__,
                           'agent_delay': NotImplemented,  # TODO: Add back later if used
                           'agent_actions_per_turn': self.sim.agent.actions_per_turn,
                           'agent_action_space_vaccinate_cost': self.reference_env.sds_env.action_space.vaccinate_cost,
                           'agent_action_space_isolate_cost': self.reference_env.sds_env.action_space.isolate_cost,
                           })  # TODO: Other action costs, etc.

        metrics_to_log = {}

        # These are already totals
        for c in ["Observed overall score", "Observed turn score", "Overall score", "Turn score", "Total deaths"]:
            metrics_to_log.update(self._agg_stats(self.results[c]))

        mlflow.log_metrics(metrics_to_log)

        mlflow.end_run()
