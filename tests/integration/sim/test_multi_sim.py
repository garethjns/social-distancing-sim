import unittest

import numpy as np
from tqdm import tqdm

from social_distancing_sim.agent.basic_agents.dummy_agent import DummyAgent
from social_distancing_sim.agent.basic_agents.isolation_agent import IsolationAgent
from social_distancing_sim.agent.basic_agents.random_agent import RandomAgent
from social_distancing_sim.agent.basic_agents.treatment_agent import TreatmentAgent
from social_distancing_sim.agent.basic_agents.vaccination_agent import VaccinationAgent
from social_distancing_sim.environment.disease import Disease
from social_distancing_sim.environment.environment import Environment
from social_distancing_sim.environment.graph import Graph
from social_distancing_sim.environment.healthcare import Healthcare
from social_distancing_sim.environment.observation_space import ObservationSpace
from social_distancing_sim.sim.multi_sim import MultiSim
from social_distancing_sim.sim.sim import Sim


class TestMultiSim(unittest.TestCase):
    @staticmethod
    def _run_with_agent(agent):
        # Arrange
        seed = None
        pop = Environment(name="example environment",
                          disease=Disease(name='COVID-19',
                                          virulence=0.01,
                                          seed=seed,
                                          immunity_mean=0.95,
                                          immunity_decay_mean=0.05),
                          healthcare=Healthcare(capacity=5),
                          observation_space=ObservationSpace(graph=Graph(community_n=5,
                                                                         community_size_mean=5,
                                                                         seed=seed),
                                                             test_rate=1,
                                                             seed=seed),
                          seed=seed)

        multi_sims = []
        for n_actions, agent in np.array(np.meshgrid([5, 10], [agent])).T.reshape(-1, 2):
            print(agent)
            multi_sims.append(MultiSim(Sim(env=pop,
                                           n_steps=50,
                                           agent=agent(env=pop,
                                                       actions_per_turn=n_actions,
                                                       seed=seed)),
                                       name='basic agent comparison',
                                       n_reps=3,
                                       n_jobs=1))

        # Act
        for ms in tqdm(multi_sims):
            ms.run()

    def test_multi_sim_run_with_isolation_agent(self):
        self._run_with_agent(IsolationAgent)

    def test_multi_sim_run_with_random_agent(self):
        self._run_with_agent(RandomAgent)

    def test_multi_sim_run_with_dummy_agent(self):
        self._run_with_agent(DummyAgent)

    def test_multi_sim_run_with_treatment_agent(self):
        self._run_with_agent(TreatmentAgent)

    def test_multi_sim_run_with_vaccination_agent(self):
        self._run_with_agent(VaccinationAgent)
