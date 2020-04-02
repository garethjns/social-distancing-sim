from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.population.graph import Graph
from social_distancing_sim.population.healthcare import Healthcare
from social_distancing_sim.population.observation_space import ObservationSpace
from social_distancing_sim.population.population import Population

if __name__ == "__main__":
    disease = Disease(name='COVID-19')
    healthcare = Healthcare()

    pop = Population(name='A herd of cats',
                     disease=disease,
                     healthcare=healthcare,
                     observation_space=ObservationSpace(graph=Graph(community_n=40,
                                                                    community_size_mean=16,
                                                                    seed=123),
                                                        test_rate=1))

    pop_distanced = Population(name='A socially responsible population',
                               disease=disease,
                               healthcare=healthcare,
                               observation_space=ObservationSpace(graph=Graph(community_n=40,
                                                                              community_size_mean=16,
                                                                              community_p_in=0.05,
                                                                              community_p_out=0.04,
                                                                              seed=123),
                                                                  test_rate=1))

    pop_distanced.run(steps=130,
                      plot=False)
    pop_distanced.replay(duration=0.1)

    pop.run(steps=130,
            plot=False)
    pop.replay(duration=0.1)
