import unittest

from joblib import Parallel, delayed

from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.population.graph import Graph
from social_distancing_sim.population.healthcare import Healthcare
from social_distancing_sim.population.observation_space import ObservationSpace
from social_distancing_sim.population.population import Population


def run_and_replay(pop, *args, **kwargs):
    pop.run(*args, **kwargs)


class TestPopulation(unittest.TestCase):
    def test_population_run(self):
        graph = Graph(community_n=15,
                      community_size_mean=5)

        pop = Population(name="example population",
                         disease=Disease(name='COVID-19'),
                         healthcare=Healthcare(),
                         observation_space=ObservationSpace(graph=graph,
                                                            test_rate=1),
                         plot_ts_fields_g2=["Score"],
                         plot_ts_obs_fields_g2=["Observed score"])

        pop.run(steps=50,
                plot=False,
                save=False)

    def test_population_run_with_testing_rate_lt_1(self):
        graph = Graph(community_n=15,
                      community_size_mean=5)

        pop = Population(name="example population",
                         disease=Disease(name='COVID-19'),
                         healthcare=Healthcare(),
                         observation_space=ObservationSpace(graph=graph,
                                                            test_rate=0.2),
                         plot_ts_fields_g2=["Score"],
                         plot_ts_obs_fields_g2=["Observed score"])

        pop.run(steps=50,
                plot=False,
                save=False)

    def test_compare_two_pops(self):
        disease = Disease(name='COVID-19')
        healthcare = Healthcare()

        pop_close = Population(name='A herd of cats, observed',
                               disease=disease,
                               healthcare=healthcare,
                               observation_space=ObservationSpace(graph=Graph(community_n=15,
                                                                              community_size_mean=10,
                                                                              seed=123),
                                                                  test_rate=0.1))

        pop_distanced = Population(name='A socially responsible population, observed',
                                   disease=disease,
                                   healthcare=healthcare,
                                   observation_space=ObservationSpace(graph=Graph(community_n=15,
                                                                                  community_size_mean=10,
                                                                                  community_p_in=0.05,
                                                                                  community_p_out=0.04,
                                                                                  seed=123),
                                                                      test_rate=0.2))

        Parallel(n_jobs=1,
                 backend='loky')(delayed(run_and_replay)(pop,
                                                         steps=50,
                                                         plot=False,
                                                         save=False) for pop in [pop_close, pop_distanced])
