import glob
import os
import tempfile
import unittest

import gym

from social_distancing_sim.agent.basic_agents.vaccination_agent import VaccinationAgent
from social_distancing_sim.sim.sim import Sim
from tests.common.env_fixtures import register_sim_test_envs


class TestSim(unittest.TestCase):
    _test_field = 'Turn score'
    _sut: Sim = Sim

    def setUp(self):
        register_sim_test_envs()

        self._tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        try:
            self._tmp_dir.cleanup()
        except PermissionError:
            # Windows.......
            pass

    def test_default_sim_run(self):
        # Arrange
        sim = self._sut(env_spec=gym.make('SDSTests-GymEnvDefaultFixture-v0').spec,
                        n_steps=10,
                        agent=VaccinationAgent(),
                        plot=False, save=False)

        # Act
        history = sim.run()

        # Assert
        self.assertEqual(10, len(history[self._test_field]))
        self.assertEqual(10, sim._step)
        self.assertEqual(10, len(sim.env.sds_env.history[self._test_field]))

    def test_example_sim_run(self):
        # Arrange
        sim = self._sut(env_spec=gym.make('SDSTests-GymEnvDefaultFixture-v0').spec,
                        save_dir=f"{self._tmp_dir.name}",
                        agent=VaccinationAgent(actions_per_turn=25, seed=123),
                        plot=False, save=False)

        # Act
        history = sim.run()

        # Assert
        self.assertEqual(100, len(history[self._test_field]))
        self.assertEqual(100, sim._step)
        self.assertEqual(100, len(sim.env.sds_env.history[self._test_field]))

    def test_example_sim_run_with_plotting(self):
        # Arrange
        sim = self._sut(env_spec=gym.make('SDSTests-GymEnvSomePlottingFixture-v0').spec,
                        save_dir=f"{self._tmp_dir.name}",
                        agent=VaccinationAgent(actions_per_turn=25, seed=123),
                        plot=False, save=True, n_steps=12)

        # Act
        history = sim.run()
        sim.env.sds_env.replay()

        # Assert
        self.assertEqual(12, len(history[self._test_field]))
        self.assertEqual(12, sim._step)
        self.assertEqual(12, len(sim.env.sds_env.history[self._test_field]))
        self.assertEqual(12 + 1, len(glob.glob(os.path.join(sim.env.sds_env.environment_plotting.graph_path, "*.png"))))

    def test_example_sim_run_with_extra_plotting(self):
        # Arrange
        sim = self._sut(env_spec=gym.make('SDSTests-GymEnvExtraPlottingFixture-v0').spec,
                        save_dir=f"{self._tmp_dir.name}",
                        agent=VaccinationAgent(actions_per_turn=25, seed=123),
                        plot=False, save=True, n_steps=16)

        # Act
        history = sim.run()
        sim.env.sds_env.replay()

        # Assert
        self.assertEqual(16, len(history[self._test_field]))
        self.assertEqual(16, sim._step)
        self.assertEqual(16, len(sim.env.sds_env.history[self._test_field]))
        self.assertEqual(16 + 1, len(glob.glob(os.path.join(sim.env.sds_env.environment_plotting.graph_path, "*.png"))))
