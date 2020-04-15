"""Run a single environment for 2 years with imperfect immunity and limited test rate."""

from joblib import Parallel, delayed

import social_distancing_sim.environment as env


def run_and_replay(pop, *args, **kwargs):
    pop.run(*args, **kwargs)
    if save:
        pop.replay(duration=duration)


if __name__ == "__main__":
    save = True
    duration = 0.1

    pop_low_immunity = env.Environment(name="Low immunity environment (small)",
                                       disease=env.Disease(name='COVID-19',
                                                           virulence=0.035,
                                                           recovery_rate=0.99,
                                                           immunity_mean=0.66,
                                                           immunity_decay_mean=0.1,
                                                           seed=125),
                                       environment_plotting=env.EnvironmentPlotting(
                                           ts_fields_g2=["Mean immunity (of immune nodes)",
                                                         "Mean immunity (of all alive nodes)"],
                                           ts_obs_fields_g2=[
                                               "Known mean immunity (of immune nodes)",
                                               "Known mean immunity (of all alive nodes)"]),
                                       observation_space=env.ObservationSpace(graph=env.Graph(community_n=8,
                                                                                              community_size_mean=5,
                                                                                              community_p_in=0.2,
                                                                                              community_p_out=0.1,
                                                                                              seed=200),
                                                                              test_rate=0.05,
                                                                              seed=123),
                                       healthcare=env.Healthcare(capacity=300),
                                       seed=124)

    pop_high_immunity = env.Environment(name="High immunity environment (small)",
                                        disease=env.Disease(name='COVID-19',
                                                            virulence=0.035,
                                                            recovery_rate=0.99,
                                                            immunity_mean=0.95,
                                                            immunity_decay_mean=0.02,
                                                            seed=125),
                                        environment_plotting=env.EnvironmentPlotting(
                                            ts_fields_g2=["Mean immunity (of immune nodes)",
                                                          "Mean immunity (of all alive nodes)"],
                                            ts_obs_fields_g2=[
                                                "Known mean immunity (of immune nodes)",
                                                "Known mean immunity (of all alive nodes)"]),
                                        observation_space=env.ObservationSpace(graph=env.Graph(community_n=8,
                                                                                               community_size_mean=5,
                                                                                               community_p_in=0.2,
                                                                                               community_p_out=0.1,
                                                                                               seed=200),
                                                                               test_rate=0.05,
                                                                               seed=123),
                                        healthcare=env.Healthcare(capacity=300),
                                        seed=124)

    Parallel(n_jobs=2,
             backend='loky')(delayed(run_and_replay)(pop,
                                                     steps=365,
                                                     plot=False,
                                                     save=True) for pop in [pop_low_immunity, pop_high_immunity])
