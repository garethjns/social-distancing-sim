import social_distancing_sim.environment as env
import social_distancing_sim.sim as sim

if __name__ == "__main__":
    seed = 123

    pop = env.Environment(name="agent example environment",
                          disease=env.Disease(name='COVID-19',
                                              virulence=0.01,
                                              seed=seed,
                                              immunity_mean=0.95,
                                              immunity_decay_mean=0.05),
                          healthcare=env.Healthcare(capacity=5),
                          observation_space=env.ObservationSpace(graph=env.Graph(community_n=15,
                                                                                 community_size_mean=10,
                                                                                 seed=seed + 1),
                                                                 test_rate=1,
                                                                 seed=seed + 2),
                          seed=seed + 3,
                          environment_plotting=env.EnvironmentPlotting(ts_fields_g2=["Turn score", "Action cost",
                                                                                     "Overall score"],
                                                                       ts_obs_fields_g2=["Observed turn score",
                                                                                         "Action cost",
                                                                                         "Observed overall score"]))

    sim = sim.Sim(env=pop,
                  agent=env.accinationAgent(actions_per_turn=25,
                                            seed=seed),
                  plot=True,
                  save=True)

    sim.run()
