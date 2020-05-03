import sys
import unittest
from unittest import mock

import numpy as np

from social_distancing_sim.gym.gym_env import GymEnv

sys.modules["keras"] = mock.MagicMock()
sys.modules["tensorflow"] = mock.MagicMock()
import social_distancing_sim.gym.agent.rl.q_learners.deep_q_agent as dqn


class TestDeepQAgent(unittest.TestCase):

    def setUp(self):
        mock_env = mock.MagicMock(spec=GymEnv)
        mock_env.observation_space[0].sample.return_value = np.random.randint(0, 100, size=(6,))

        self._mock_env = mock_env

        dqn.DeepQAgent._prep_models = lambda x: x
        self._sut = dqn.DeepQAgent(mock_env)

    @staticmethod
    def _mock_obs():
        return np.random.randint(0, 100, size=(6,)), np.random.binomial(1, 0.5, size=(20, 20))

    def test_init_with_defaults(self):
        self.assertIsInstance(self._sut, dqn.DeepQAgent)

    def test_transform_handles_single_row_state_tuple(self):
        # Act
        trans_state = self._sut.transform(self._mock_obs())

        # Assert
        self.assertEqual((1, 6), trans_state[0].shape)
        self.assertEqual((1, 20, 20, 1), trans_state[1].shape)

    def test_transform_handles_batch_state_tuple(self):
        # Arrange
        state = self._mock_obs()
        state_batch = (np.stack([state[0], state[0]]), np.stack([state[1], state[1]]))

        # Act
        trans_state = self._sut.transform(state_batch)

        # Assert
        self.assertEqual((2, 6), trans_state[0].shape)
        self.assertEqual((2, 20, 20, 1), trans_state[1].shape)

    def test_transform_handles_single_row_state_tuple_with_extra_dims(self):
        # Arrange
        state = self._mock_obs()
        state_expanded = (np.atleast_2d(state[0]), np.expand_dims(np.expand_dims(state[1], axis=0), axis=3))

        # Act
        trans_state = self._sut.transform(state_expanded)

        # Assert
        self.assertEqual((1, 6), trans_state[0].shape)
        self.assertEqual((1, 20, 20, 1), trans_state[1].shape)

    def test_transform_handles_batch_state_tuple_with_extra_dims(self):
        # Arrange
        state = self._mock_obs()
        state_batch = (np.stack([state[0], state[0]]), np.stack([state[1], state[1]]))
        state_expanded_batch = (state_batch[0], np.expand_dims(state_batch[1], axis=3))

        # Act
        trans_state = self._sut.transform(state_expanded_batch)

        # Assert
        self.assertEqual((2, 6), trans_state[0].shape)
        self.assertEqual((2, 20, 20, 1), trans_state[1].shape)

    def test_filter_state_handles_full_state_as_input(self):
        # Act
        state = self._sut._filter_state((0, 1, 2))

        # Assert
        self.assertEqual((0, 1), state)
