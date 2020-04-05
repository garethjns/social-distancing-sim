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
                   'observation_space': ObservationSpace(graph=Graph(community_n=50,
                                                                     community_size_mean=15,
                                                                     community_p_in=0.08,
                                                                     community_p_out=0.04),
                                                         test_rate=0.05),
                   'healthcare': Healthcare(capacity=200)}

    pop_low_immunity = Population(name="Low immunity population",
                                  disease=Disease(name='COVID-19',
                                                  virulence=0.005,
                                                  recovery_rate=0.99,
                                                  immunity_mean=0.66,
                                                  immunity_decay_mean=0.1),
                                  **common_args)

    pop_high_immunity = Population(name="High immunity population",
                                   disease=Disease(name='COVID-19',
                                                   virulence=0.005,
                                                   recovery_rate=0.99,
                                                   immunity_mean=0.98,
                                                   immunity_decay_mean=0.01),
                                   **common_args)

    Parallel(n_jobs=2,
             backend='loky')(delayed(run_and_replay)(pop,
                                                     steps=750,
                                                     plot=False,
                                                     save=True) for pop in [pop_low_immunity, pop_high_immunity])
