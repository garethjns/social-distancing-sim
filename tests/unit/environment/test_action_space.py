import unittest

from social_distancing_sim.environment.action_space import ActionSpace


class TestActionSpace(unittest.TestCase):
    _sut = ActionSpace()
    _implemented_actions = ['vaccinate', 'isolate', 'reconnect', 'treat']

    def test_expected_default_actions_are_available(self):
        # Act
        available_actions = self._sut.available_actions

        # Assert
        self.assertListEqual(self._implemented_actions, available_actions)

    def test_n_returns_expected_n_actions(self):
        # Act
        n_available_actions = self._sut.n

        # Assert
        self.assertEqual(len(self._implemented_actions), n_available_actions)

    def test_sampled_returns_valid_action(self):
        # Act
        action = self._sut.sample()

        # Assert
        self.assertIn(action, self._implemented_actions)
