from social_distancing_sim.population.population_templates import herd_of_cats, socially_responsible
from social_distancing_sim.sim.sim import Sim

herd_of_cats_visual_example = Sim(populations=[herd_of_cats],
                                  mode='visual',
                                  steps=120)
socially_responsible_visual_example = Sim(populations=[socially_responsible],
                                          mode='visual',
                                          steps=120)
herd_of_cats_stats_example = Sim(populations=[herd_of_cats],
                                 reps=20,
                                 steps=100)
socially_responsible_stats_example = Sim(populations=[socially_responsible],
                                         reps=20,
                                         steps=100)

herd_of_cats_vs_socially_responsible_stats_example = Sim(populations=[herd_of_cats, socially_responsible],
                                                         reps=20,
                                                         steps=100)
