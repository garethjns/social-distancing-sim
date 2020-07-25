import unittest

import gym
import numpy as np
from tqdm import tqdm

from social_distancing_sim.agent.basic_agents.dummy_agent import DummyAgent
from social_distancing_sim.agent.basic_agents.isolation_agent import IsolationAgent
from social_distancing_sim.agent.basic_agents.random_agent import RandomAgent
from social_distancing_sim.agent.basic_agents.treatment_agent import TreatmentAgent
from social_distancing_sim.agent.basic_agents.vaccination_agent import VaccinationAgent
from social_distancing_sim.sim.multi_sim import MultiSim
from social_distancing_sim.sim.sim import Sim
from tests.common.env_fixtures import register_test_envs


class TestMultiSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        register_test_envs()

    @staticmethod
    def _run_with_agent(agent, n_jobs: int = 1):
        # Arrange
        env_spec = gym.make('SDSTests-GymEnvRandomSeedFixture-v0').spec
        multi_sims = []
        for n_actions, agent in np.array(np.meshgrid([5, 10], [agent])).T.reshape(-1, 2):
            print(agent)
            multi_sims.append(MultiSim(Sim(env_spec=env_spec,
                                           n_steps=50,
                                           agent=agent(actions_per_turn=n_actions,
                                                       seed=None)),
                                       name='basic agent comparison',
                                       n_reps=3,
                                       n_jobs=n_jobs))

        # Act
        for ms in tqdm(multi_sims):
            ms.run()

    def test_multi_sim_run_with_isolation_agent(self):
        self._run_with_agent(IsolationAgent, n_jobs=1)

    def test_multi_sim_run_with_isolation_agent_multiple_jobs(self):
        self._run_with_agent(IsolationAgent, n_jobs=2)

    def test_multi_sim_run_with_random_agent(self):
        self._run_with_agent(RandomAgent, n_jobs=1)

    def test_multi_sim_run_with_random_agent_multiple_jobs(self):
        self._run_with_agent(RandomAgent, n_jobs=2)

    def test_multi_sim_run_with_dummy_agent(self):
        self._run_with_agent(DummyAgent, n_jobs=1)

    def test_multi_sim_run_with_dummy_agent_multiple_jobs(self):
        self._run_with_agent(DummyAgent, n_jobs=2)

    def test_multi_sim_run_with_treatment_agent(self):
        self._run_with_agent(TreatmentAgent, n_jobs=1)

    def test_multi_sim_run_with_treatment_agent_multiple_jobs(self):
        self._run_with_agent(TreatmentAgent, n_jobs=2)

    def test_multi_sim_run_with_vaccination_agent(self):
        self._run_with_agent(VaccinationAgent, n_jobs=1)

    def test_multi_sim_run_with_vaccination_agent_multiple_jobs(self):
        self._run_with_agent(VaccinationAgent, n_jobs=2)
