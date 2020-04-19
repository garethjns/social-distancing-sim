import unittest
from unittest.mock import MagicMock

from social_distancing_sim.agent.policy_agents.distancing_policy_agent import DistancingPolicyAgent


class TestDistancingPolicyAgent(unittest.TestCase):
    _sut = DistancingPolicyAgent

    def setUp(self) -> None:
        mock_observation_space = MagicMock()
        mock_observation_space.current_clear_nodes = [9, 10, 11, 12]
        mock_observation_space.isolated_nodes = [12, 13, 14]
        self.mock_observation_space = mock_observation_space

    def test_init_with_defaults(self):
        # Act
        agent = self._sut()

        # Assert
        self.assertIsInstance(agent,  DistancingPolicyAgent)

    def test_available_actions(self):
        # Arrange
        agent = self._sut()

        # Assert
        self.assertListEqual(['isolate', 'reconnect'], agent.available_actions)

    def test_no_actions_outside_active_period(self):
        # Arrange
        agent = self._sut(name='test_agent',
                          actions_per_turn=1,
                          start_step={'isolate': 25,
                                      'reconnect': 35},
                          end_step={'isolate': 30,
                                    'reconnect': 40})

        # Act
        action = agent.get_actions(self.mock_observation_space)

        # Assert
        self.assertListEqual([], list(action.keys()))

    def test_n_actions_inside_first_active_period(self):
        # Arrange
        agent = self._sut(name='test_agent',
                          actions_per_turn=1,
                          start_step={'isolate': 25,
                                      'reconnect': 35},
                          end_step={'isolate': 30,
                                    'reconnect': 40})
        agent._step = 26

        # Act
        action = agent.get_actions(self.mock_observation_space)

        # Assert
        self.assertListEqual(['isolate'], list(action.values()))

    def test_no_actions_between_active_periods(self):
        # Arrange
        agent = self._sut(name='test_agent',
                          actions_per_turn=1,
                          start_step={'isolate': 25,
                                      'reconnect': 35},
                          end_step={'isolate': 30,
                                    'reconnect': 40})
        agent._step = 32

        # Act
        action = agent.get_actions(self.mock_observation_space)

        # Assert
        self.assertListEqual([], list(action.values()))

    def test_n_actions_inside_second_active_period(self):
        # Arrange
        agent = self._sut(name='test_agent',
                          actions_per_turn=1,
                          start_step={'isolate': 25,
                                      'reconnect': 35},
                          end_step={'isolate': 30,
                                    'reconnect': 40})
        agent._step = 36

        # Act
        action = agent.get_actions(self.mock_observation_space)

        # Assert
        self.assertListEqual(['reconnect'], list(action.values()))

    def test_no_actions_after_active_periods(self):
        # Arrange
        agent = self._sut(name='test_agent',
                          actions_per_turn=1,
                          start_step={'isolate': 25,
                                      'reconnect': 35},
                          end_step={'isolate': 30,
                                    'reconnect': 40})
        agent._step = 45

        # Act
        action = agent.get_actions(self.mock_observation_space)

        # Assert
        self.assertListEqual([], list(action.values()))

    def test_whole_active_period_returns_actions_with_single_actions(self):
        # Arrange
        agent = self._sut(name='test_agent',
                          actions_per_turn=1,
                          start_step={'isolate': 5,
                                      'reconnect': 12},
                          end_step={'isolate': 10,
                                    'reconnect': 16})

        # Act
        actions = []
        for s in range(20):
            actions.append(agent.get_actions(self.mock_observation_space))

        # Assert
        self.assertEqual(20, len(actions))
        self.assertEqual(20, agent._step)
        self.assertListEqual([0, 0, 0, 0, 0,
                              1, 1, 1, 1, 1, 1,
                              0,
                              1, 1, 1, 1, 1,
                              0, 0, 0], [len(d.keys()) for d in actions])

    def test_whole_active_period_returns_actions_with_multiple_actions(self):
        # Arrange
        agent = self._sut(name='test_agent',
                          actions_per_turn=3,
                          start_step={'isolate': 5,
                                      'reconnect': 12},
                          end_step={'isolate': 10,
                                    'reconnect': 16})

        # Act
        actions = []
        for s in range(20):
            actions.append(agent.get_actions(self.mock_observation_space))

        # Assert
        self.assertEqual(20, len(actions))
        self.assertEqual(20, agent._step)
        # Likely to be fewer than 3 actions if duplicate target selected from small pool.
        self.assertListEqual([False, False, False, False, False,
                              True, True, True, True, True, True,
                              False,
                              True, True, True, True, True,
                              False, False, False], [len(d.keys()) > 0 for d in actions])
