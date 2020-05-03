import shutil
import unittest

from social_distancing_sim.agent.basic_agents.vaccination_agent import VaccinationAgent
from social_distancing_sim.environment.disease import Disease
from social_distancing_sim.environment.environment import Environment
from social_distancing_sim.environment.environment_plotting import EnvironmentPlotting
from social_distancing_sim.environment.graph import Graph
from social_distancing_sim.environment.healthcare import Healthcare
from social_distancing_sim.environment.observation_space import ObservationSpace
from social_distancing_sim.sim.sim import Sim


class TestSim(unittest.TestCase):

    def setUp(self):
        self._to_delete = None

        seed = 123
        self._common_setup = {'disease': Disease(name='COVID-19',
                                                 virulence=0.01,
                                                 seed=seed,
                                                 immunity_mean=0.95,
                                                 immunity_decay_mean=0.05),
                              'healthcare': Healthcare(capacity=5),
                              'observation_space': ObservationSpace(graph=Graph(community_n=15,
                                                                                community_size_mean=10,
                                                                                seed=seed + 1),
                                                                    test_rate=1,
                                                                    seed=seed + 2),
                              'seed': seed + 3}
        self._ts_fields_g2 = ["Score", "Action cost",
                              "Overall score"]
        self._ts_obs_fields_g2 = ["Observed Score",
                                  "Action cost",
                                  "Observed overall score"]

    def tearDown(self):
        if self._to_delete is not None:
            shutil.rmtree(self._to_delete, ignore_errors=True)

    def test_default_sim_run(self):
        pop = Environment(disease=Disease(),
                          healthcare=Healthcare(),
                          observation_space=ObservationSpace(graph=Graph()))

        sim = Sim(env=pop,
                  agent=VaccinationAgent(env=pop),
                  plot=False,
                  save=False)

        sim.run()

    def test_example_sim_run(self):
        pop = Environment(name="agent example environment 1",
                          environment_plotting=EnvironmentPlotting(),
                          **self._common_setup)

        sim = Sim(env=pop,
                  agent=VaccinationAgent(env=pop,
                                         actions_per_turn=25,
                                         seed=self._common_setup["seed"]),
                  plot=False,
                  save=False)

        sim.run()

        self._to_delete = pop.name

    def test_example_sim_run_with_plotting(self):
        seed = 123

        pop = Environment(name="agent example environment 2",
                          environment_plotting=EnvironmentPlotting(ts_fields_g2=self._ts_fields_g2),
                          **self._common_setup)

        sim = Sim(env=pop,
                  n_steps=3,
                  agent=VaccinationAgent(env=pop,
                                         actions_per_turn=25,
                                         seed=seed),
                  plot=False,
                  save=True)

        sim.run()
        sim.env.replay()
        self._to_delete = pop.name

    def test_example_sim_run_with_extra_plotting(self):
        seed = 123

        pop = Environment(name="agent example environment 3",
                          environment_plotting=EnvironmentPlotting(ts_fields_g2=self._ts_fields_g2,
                                                                   ts_obs_fields_g2=self._ts_obs_fields_g2),
                          **self._common_setup)

        sim = Sim(env=pop,
                  n_steps=3,
                  agent=VaccinationAgent(env=pop,
                                         actions_per_turn=25,
                                         seed=seed),
                  plot=False,
                  save=True)

        sim.run()
        sim.env.replay()
        self._to_delete = pop.name

    def test_example_sim_run_with_all_plotting(self):
        seed = 123

        pop = Environment(name="agent example environment 4",
                          environment_plotting=EnvironmentPlotting(ts_fields_g2=self._ts_fields_g2,
                                                                   ts_obs_fields_g2=self._ts_obs_fields_g2),
                          **self._common_setup)

        sim = Sim(env=pop,
                  n_steps=3,
                  agent=VaccinationAgent(env=pop,
                                         actions_per_turn=25,
                                         seed=seed),
                  plot=False,
                  save=True)

        sim.run()
        sim.env.replay()
        self._to_delete = pop.name
