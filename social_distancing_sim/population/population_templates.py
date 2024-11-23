from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.population.population import Population

herd_of_cats = Population(
    name="A herd of cats",
    disease=Disease(name="COVID-19"),
    community_n=40,
    community_size_mean=16,
    seed=123,
)

socially_responsible = Population(
    name="A socially responsible population",
    disease=Disease(name="COVID-19"),
    community_n=40,
    community_size_mean=16,
    community_p_in=0.05,
    community_p_out=0.04,
    seed=123,
)
