import unittest

from social_distancing_sim.environment.action_space import ActionSpace


class TestActionSpace(unittest.TestCase):
    _sut = ActionSpace()
    _implemented_actions = ['nothing', 'vaccinate', 'isolate', 'reconnect', 'treat']
    _implemented_action_ids = [0, 1, 2, 3, 4]

    def test_expected_default_actions_are_available(self):
        # Act
        available_actions = self._sut.available_actions

        # Assert
        self.assertListEqual(self._implemented_action_ids, available_actions)

    def test_n_returns_expected_n_actions(self):
        # Act
        n_available_actions = self._sut.n

        # Assert
        self.assertEqual(len(self._implemented_actions), n_available_actions)

    def test_sampled_returns_valid_action(self):
        # Act
        action = self._sut.sample()

        # Assert
        self.assertIn(action, self._implemented_action_ids)

    def test_name_to_id(self):
        # Act
        action_id = self._sut.get_action_id('isolate')

        # Assert
        self.assertEqual(self._implemented_actions.index('isolate'), action_id)

    def test_id_to_name(self):
        # Act
        action_name = self._sut.get_action_name(3)

        # Assert
        self.assertEqual(action_name, self._implemented_actions[3])

    def test_select_n_random_actions_with_larger_pool(self):
        # Act
        targets = self._sut.select_random_target(n=2,
                                                 available_targets=[1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Assert
        self.assertEqual(2, len(targets))
        self.assertNotIn(-1, targets)

    def test_select_n_random_actions_with_lesser_pool(self):
        # Act
        targets = self._sut.select_random_target(n=12,
                                                 available_targets=[1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Assert
        self.assertEqual(12, len(targets))
        self.assertEqual(9, len([t for t in targets if t != -1]))
        self.assertEqual(3, len([t for t in targets if t == -1]))
