from joblib import Parallel, delayed

import social_distancing_sim.environment as env


def run_and_replay(pop, *args, **kwargs):
    pop.run(*args, **kwargs)
    if save:
        pop.replay()


if __name__ == "__main__":
    save = True

    # Create a population with high inter and intra connectivity
    pop = env.Environment(name='A herd of cats',
                          disease=env.Disease(name='COVID-19'),
                          observation_space=env.ObservationSpace(graph=env.Graph(community_n=40,
                                                                                 community_size_mean=16,
                                                                                 seed=123),
                                                                 test_rate=1),
                          environment_plotting=env.EnvironmentPlotting(ts_fields_g2=["Turn score"]))

    # Create a population with reduced inter and intra connectivity
    pop_distanced = env.Environment(name='A socially responsible environment',
                                    disease=env.Disease(name='COVID-19'),
                                    observation_space=env.ObservationSpace(graph=env.Graph(community_n=40,
                                                                                           community_size_mean=16,
                                                                                           community_p_in=0.05,
                                                                                           community_p_out=0.04,
                                                                                           seed=123),
                                                                           test_rate=1),
                                    environment_plotting=env.EnvironmentPlotting(ts_fields_g2=["Turn score"]))

    # Run and save both simulations in parallel
    Parallel(n_jobs=2,
             backend='loky')(delayed(run_and_replay)(pop,
                                                     steps=300,
                                                     plot=False,
                                                     save=save) for pop in [pop, pop_distanced])
