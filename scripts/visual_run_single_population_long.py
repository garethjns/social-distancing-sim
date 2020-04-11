"""Run a single population for 2 years with imperfect immunity and limited test rate."""
from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.population.graph import Graph
from social_distancing_sim.population.healthcare import Healthcare
from social_distancing_sim.population.observation_space import ObservationSpace
from social_distancing_sim.population.population import Population

if __name__ == "__main__":
    save = True

    graph = Graph(community_n=50,
                  community_size_mean=15,
                  community_p_in=0.1,
                  community_p_out=0.05)

    pop = Population(name="example population",
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

    # Save .gif to './example population/replay.gif'
    if save:
        pop.replay()
