import social_distancing_sim.environment as env

if __name__ == "__main__":
    save = True

    pop = env.Environment(name='A herd of cats, observed',
                          disease=env.Disease(name='COVID-19'),
                          healthcare=env.healthcare(),
                          observation_space=env.ObservationSpace(graph=env.Graph(community_n=40,
                                                                                 community_size_mean=16,
                                                                                 seed=123),
                                                                 test_rate=0.01))

    pop.run(steps=130,
            plot=False)

    if save:
        pop.replay(duration=0.1)
