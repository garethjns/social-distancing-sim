import unittest

from social_distancing_sim.agent.vaccination_agent import VaccinationAgent
from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.population.graph import Graph
from social_distancing_sim.population.healthcare import Healthcare
from social_distancing_sim.population.observation_space import ObservationSpace
from social_distancing_sim.population.population import Population
from social_distancing_sim.sim.sim import Sim


class TestSim(unittest.TestCase):
    def test_default_sim_run(self):
        pop = Population(disease=Disease(),
                         healthcare=Healthcare(),
                         observation_space=ObservationSpace(graph=Graph()))

        sim = Sim(pop=pop,
                  agent_delay=10,
                  agent=VaccinationAgent(),
                  plot=False,
                  save=False)

        sim.run()

    def test_example_sim_run(self):
        seed = 123

        pop = Population(name="agent example population",
                         disease=Disease(name='COVID-19',
                                         virulence=0.01,
                                         seed=seed,
                                         immunity_mean=0.95,
                                         immunity_decay_mean=0.05),
                         healthcare=Healthcare(capacity=5),
                         observation_space=ObservationSpace(graph=Graph(community_n=15,
                                                                        community_size_mean=10,
                                                                        seed=seed + 1),
                                                            test_rate=1,
                                                            seed=seed + 2),
                         seed=seed + 3,
                         plot_ts_fields_g2=["Score", "Action cost", "Overall score"],
                         plot_ts_obs_fields_g2=["Observed Score", "Action cost", "Observed overall score"])

        sim = Sim(pop=pop,
                  agent_delay=10,
                  agent=VaccinationAgent(actions_per_turn=25,
                                         seed=seed),
                  plot=False,
                  save=False)

        sim.run()
