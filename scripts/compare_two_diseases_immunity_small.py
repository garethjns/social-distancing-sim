"""Run a single population for 2 years with imperfect immunity and limited test rate."""

from joblib import Parallel, delayed

from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.population.graph import Graph
from social_distancing_sim.population.healthcare import Healthcare
from social_distancing_sim.population.observation_space import ObservationSpace
from social_distancing_sim.population.population import Population


def run_and_replay(pop, *args, **kwargs):
    pop.run(*args, **kwargs)
    if save:
        pop.replay(duration=duration)


if __name__ == "__main__":
    save = True
    duration = 0.1

    common_args = {'plot_ts_fields_g2': ["Mean immunity (of immune nodes)",
                                         "Mean immunity (of all alive nodes)"],
                   'plot_ts_obs_fields_g2': ["Known mean immunity (of immune nodes)",
                                             "Known mean immunity (of all alive nodes)"],
                   'observation_space': ObservationSpace(graph=Graph(community_n=8,
                                                                     community_size_mean=5,
                                                                     community_p_in=0.2,
                                                                     community_p_out=0.1,
                                                                     seed=200),
                                                         test_rate=0.05,
                                                         seed=123),
                   'healthcare': Healthcare(capacity=300),
                   'seed': 124}

    pop_low_immunity = Population(name="Low immunity population (small)",
                                  disease=Disease(name='COVID-19',
                                                  virulence=0.035,
                                                  recovery_rate=0.99,
                                                  immunity_mean=0.66,
                                                  immunity_decay_mean=0.1,
                                                  seed=125),
                                  **common_args)

    pop_high_immunity = Population(name="High immunity population (small)",
                                   disease=Disease(name='COVID-19',
                                                   virulence=0.035,
                                                   recovery_rate=0.99,
                                                   immunity_mean=0.95,
                                                   immunity_decay_mean=0.02,
                                                   seed=125),
                                   **common_args)

    Parallel(n_jobs=2,
             backend='loky')(delayed(run_and_replay)(pop,
                                                     steps=365,
                                                     plot=False,
                                                     save=True) for pop in [pop_low_immunity, pop_high_immunity])
