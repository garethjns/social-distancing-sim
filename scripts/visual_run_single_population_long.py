"""Run a single environment for 2 years with imperfect immunity and limited test rate."""
from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.environment.environment import Environment
from social_distancing_sim.environment.graph import Graph
from social_distancing_sim.environment.healthcare import Healthcare
from social_distancing_sim.environment.observation_space import ObservationSpace

if __name__ == "__main__":
    save = True

    graph = Graph(community_n=50,
                  community_size_mean=15,
                  community_p_in=0.1,
                  community_p_out=0.05)

    pop = Environment(name="example environment",
                      disease=Disease(name='COVID-19',
                                      virulence=0.006,
                                      immunity_mean=0.6,
                                      immunity_decay_mean=0.15),
                      healthcare=Healthcare(),
                      observation_space=ObservationSpace(graph=graph,
                                                         test_rate=0.05))

    pop.run(steps=750,
            plot=True,
            save=True)

    # Save .gif to './example environment/replay.gif'
    if save:
        pop.replay()
