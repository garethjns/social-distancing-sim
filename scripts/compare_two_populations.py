from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.population.population import Population

if __name__ == "__main__":
    disease = Disease(name='COVID-19')

    pop = Population(name='A herd of cats',
                     disease=disease,
                     community_n=40,
                     community_size_mean=16,
                     seed=123)

    pop_distanced = Population(name='A socially responsible population',
                               disease=disease,
                               community_n=40,
                               community_size_mean=16,
                               community_p_in=0.05,
                               community_p_out=0.04,
                               seed=123)

    pop_distanced.run(steps=150,
                      plot=False)
    pop.run(steps=150,
            plot=False)

    pop_distanced.replay(duration=0.1)
    pop.replay(duration=0.1)
