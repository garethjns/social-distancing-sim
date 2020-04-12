from joblib import Parallel, delayed

from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.environment.environment import Environment
from social_distancing_sim.environment.graph import Graph
from social_distancing_sim.environment.healthcare import Healthcare
from social_distancing_sim.environment.observation_space import ObservationSpace


def run_and_replay(pop, *args, **kwargs):
    pop.run(*args, **kwargs)
    if save:
        pop.replay()


if __name__ == "__main__":
    save = True
    disease = Disease(name='COVID-19')
    healthcare = Healthcare()

    pop_close = Environment(name='A herd of cats, observed',
                            disease=disease,
                            healthcare=healthcare,
                            observation_space=ObservationSpace(graph=Graph(community_n=40,
                                                                           community_size_mean=16,
                                                                           seed=123),
                                                               test_rate=0.04))

    pop_distanced = Environment(name='A socially responsible environment, observed',
                                disease=disease,
                                healthcare=healthcare,
                                observation_space=ObservationSpace(graph=Graph(community_n=40,
                                                                               community_size_mean=16,
                                                                               community_p_in=0.05,
                                                                               community_p_out=0.04,
                                                                               seed=123),
                                                                   test_rate=0.04))

    Parallel(n_jobs=2,
             backend='loky')(delayed(run_and_replay)(pop,
                                                     steps=130,
                                                     plot=False,
                                                     save=save) for pop in [pop_close, pop_distanced])
