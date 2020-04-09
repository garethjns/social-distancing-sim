from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.population.graph import Graph
from social_distancing_sim.population.healthcare import Healthcare
from social_distancing_sim.population.observation_space import ObservationSpace
from social_distancing_sim.population.population import Population

if __name__ == "__main__":
    save = True

    disease = Disease(name='COVID-19')
    healthcare = Healthcare()

    pop = Population(name='A herd of cats, observed',
                     disease=disease,
                     healthcare=healthcare,
                     observation_space=ObservationSpace(graph=Graph(community_n=40,
                                                                    community_size_mean=16,
                                                                    seed=123),
                                                        test_rate=0.01))

    pop.run(steps=130,
            plot=False)
    if save:
        pop.replay(duration=0.1)
