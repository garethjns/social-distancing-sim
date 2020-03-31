from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.population.population import Population

if __name__ == "__main__":
    pop = Population(name="example population",
                     disease=Disease(name='COVID-19'),
                     community_n=50,
                     community_size_mean=15,
                     healthcare_test_rate=0.01)

    pop.run(steps=100)

    # Save .gif to './example population/replay.gif'
    pop.replay()
