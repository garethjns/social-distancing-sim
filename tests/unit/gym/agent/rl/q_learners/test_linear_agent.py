import unittest
from unittest import mock

import numpy as np

from social_distancing_sim.gym.agent.rl.q_learners.linear_q_agent import LinearQAgent
from social_distancing_sim.gym.gym_env import GymEnv


class TestLinearQAgent(unittest.TestCase):
    # TODO
    _sut = LinearQAgent

    def setUp(self):
        mock_env = mock.MagicMock(spec=GymEnv)
        mock_env.observation_space[0].sample.return_value = np.random.randint(0, 100, size=(6,))
        mock_env.reset.return_value = np.random.randint(0, 100, size=(6,))
        self._mock_env = mock_env

    def test_init_with_agent_base_options_specified(self):
        agent = self._sut(env=self._mock_env, name='testq', actions_per_turn=18)

        self.assertEqual('testq', agent.name)
        self.assertEqual(18, agent.actions_per_turn)

    def test_filter_state_handles_full_state_as_input(self):
        # Act
        state = self._sut._filter_state((0, 1, 2))

        # Assert
        self.assertEqual(0, state)
