"""Run a single population with perfrect testing."""

from social_distancing_sim.population.graph import Graph
from social_distancing_sim.population.healthcare import Healthcare
from social_distancing_sim.population.observation_space import ObservationSpace
from social_distancing_sim.population.population import Population
from social_distancing_sim.disease.disease import Disease

if __name__ == "__main__":
    save = True

    graph = Graph(community_n=50,
                  community_size_mean=15)

    pop = Population(name="example population",
                     disease=Disease(name='COVID-19'),
                     healthcare=Healthcare(),
                     observation_space=ObservationSpace(graph=graph,
                                                        test_rate=1))

    pop.run(steps=50,
            plot=True,
            save=True)

    # Save .gif to './example population/replay.gif'
    if save:
        pop.replay()
