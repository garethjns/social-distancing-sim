import unittest
from unittest import mock

from social_distancing_sim.environment.environment import Environment
from social_distancing_sim.environment.observation_space import ObservationSpace
from social_distancing_sim.environment.action_space import ActionSpace


class TestEnvironment(unittest.TestCase):
    _sut = Environment

    def setUp(self):
        mock_env = mock.MagicMock(spec=self._sut)
        mock_env.logger = mock.MagicMock()
        mock_env.select_reasonable_targets = self._sut.select_reasonable_targets
        mock_env.observation_space = mock.MagicMock(sepc=ObservationSpace)
        mock_env.action_space = mock.MagicMock(sepc=ActionSpace)
        mock_env.action_space.select_random_target = ActionSpace.select_random_target
        mock_env.observation_space.current_clear_nodes = [1, 2, 3, 10, 11]
        mock_env.observation_space.current_infected_nodes = [4, 5, 6, 12]
        mock_env.observation_space.current_immune_nodes = [7, 8, 9]
        mock_env.observation_space.current_isolated_nodes = [10, 11, 12]

        self.mock_env = mock_env

    def test_select_random_targets_for_a_selection_of_valid_actions(self):
        # NB: Action 4 can replace action 2 with default environments selections
        actions_dict = self.mock_env.select_reasonable_targets(self=self.mock_env, actions=[0, 1, 3, 4, 4, 4])

        # Assert
        # Inexact - action selections are partly random so can overwrite previously selected targets
        self.assertGreater(len(actions_dict.keys()), 3)
        self.assertGreater(len([v for v in actions_dict.values() if v == 4]), 1)
        self.assertNotIn(0, actions_dict.keys())
        self.assertNotIn(-1, actions_dict.keys())

    def test_select_random_targets_for_another_selection_of_valid_actions(self):
        actions_dict = self.mock_env.select_reasonable_targets(self=self.mock_env, actions=[0, 1, 2, 2, 3])

        # Assert
        # Inexact - action selections are partly random so can overwrite previously selected targets
        self.assertGreater(len(actions_dict.keys()), 2)
        self.assertGreater(len([v for v in actions_dict.values() if v == 2]), 0)
        self.assertNotIn(0, actions_dict.keys())
        self.assertNotIn(-1, actions_dict.keys())

    def test_select_random_targets_for_more_valid_actions_than_available_targets(self):
        actions_dict = self.mock_env.select_reasonable_targets(self=self.mock_env, actions=[0, 1, 2, 3, 4, 4, 4, 4, 4])

        # Assert
        # Inexact - action selections are partly random so can overwrite previously selected targets
        self.assertGreater(len(actions_dict.keys()), 4)
        self.assertGreater(len([v for v in actions_dict.values() if v == 4]), 2)
        self.assertNotIn(0, actions_dict.keys())
        self.assertNotIn(-1, actions_dict.keys())
