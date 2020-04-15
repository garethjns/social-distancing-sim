"""Run a single environment for 2 years with imperfect immunity and limited test rate."""

import social_distancing_sim.environment as env

if __name__ == "__main__":
    save = True

    graph = env.Graph(community_n=50,
                      community_size_mean=15,
                      community_p_in=0.1,
                      community_p_out=0.05)

    pop = env.Environment(name="example environment",
                          disease=env.Disease(name='COVID-19',
                                              virulence=0.006,
                                              immunity_mean=0.6,
                                              immunity_decay_mean=0.15),
                          healthcare=env.Healthcare(),
                          observation_space=env.ObservationSpace(graph=graph,
                                                                 test_rate=0.05))

    pop.run(steps=750,
            plot=True,
            save=True)

    # Save .gif to './example environment/replay.gif'
    if save:
        pop.replay()
