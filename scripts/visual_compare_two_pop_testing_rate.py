from joblib import Parallel, delayed

import social_distancing_sim.environment as env


def run_and_replay(pop, *args, **kwargs):
    pop.run(*args, **kwargs)
    if save:
        pop.replay()


if __name__ == "__main__":
    save = True

    # Define the populations used in the social distancing comparison, but specify a test rate of 4% / per turn for
    # the observation space. An plot will be added showing the data the observation space can see.
    pop_close = env.Environment(name='A herd of cats, observed',
                                disease=env.Disease(name='COVID-19'),
                                observation_space=env.ObservationSpace(graph=env.Graph(community_n=40,
                                                                                       community_size_mean=16,
                                                                                       seed=123),
                                                                       test_rate=0.04))

    pop_distanced = env.Environment(name='A socially responsible environment, observed',
                                    disease=env.Disease(name='COVID-19'),
                                    observation_space=env.ObservationSpace(graph=env.Graph(community_n=40,
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
