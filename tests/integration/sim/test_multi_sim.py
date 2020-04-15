import unittest

import numpy as np
from tqdm import tqdm

from social_distancing_sim.agent.isolation_agent import IsolationAgent
from social_distancing_sim.agent.vaccination_agent import VaccinationAgent
from social_distancing_sim.environment.disease import Disease
from social_distancing_sim.environment.environment import Environment
from social_distancing_sim.environment.graph import Graph
from social_distancing_sim.environment.healthcare import Healthcare
from social_distancing_sim.environment.observation_space import ObservationSpace
from social_distancing_sim.sim.multi_sim import MultiSim
from social_distancing_sim.sim.sim import Sim


class TestMultiSim(unittest.TestCase):
    def test_multi_sim_run(self):
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
        for n_actions, agent in np.array(np.meshgrid([5, 10], [VaccinationAgent, IsolationAgent])).T.reshape(-1, 2):
            print(agent)
            multi_sims.append(MultiSim(Sim(env=pop,
                                           n_steps=50,
                                           agent=agent(actions_per_turn=n_actions,
                                                       seed=seed)),
                                       name='basic agent comparison',
                                       n_reps=3,
                                       n_jobs=1))

        # Act
        for ms in tqdm(multi_sims):
            ms.run()
