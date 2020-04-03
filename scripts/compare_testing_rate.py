from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.population.graph import Graph
from social_distancing_sim.population.healthcare import Healthcare
from social_distancing_sim.population.observation_space import ObservationSpace
from social_distancing_sim.population.population import Population

if __name__ == "__main__":
    disease = Disease(name='COVID-19')
    healthcare = Healthcare()

    pop_full_knowledge = Population(name='A herd of cats, reality',
                                    disease=disease,
                                    healthcare=healthcare,
                                    observation_space=ObservationSpace(graph=Graph(community_n=40,
                                                                                   community_size_mean=16,
                                                                                   seed=123),
                                                                       test_rate=1))
    pop_poor_testing_knowledge = Population(name='A herd of cats, observed',
                                            disease=disease,
                                            healthcare=healthcare,
                                            observation_space=ObservationSpace(graph=Graph(community_n=40,
                                                                                           community_size_mean=16,
                                                                                           seed=123),
                                                                               test_rate=0.01))

    pop_full_knowledge.run(steps=130,
                           plot=False)
    pop_full_knowledge.replay(duration=0.1)

    pop_poor_testing_knowledge.run(steps=130,
                                   plot=False)
    pop_poor_testing_knowledge.replay(duration=0.1)
