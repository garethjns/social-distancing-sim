import unittest
from unittest.mock import MagicMock

from social_distancing_sim.agent.policy_agents.vaccination_policy_agent import VaccinationPolicyAgent


class TestVaccinationPolicyAgent(unittest.TestCase):
    _sut = VaccinationPolicyAgent

    def setUp(self) -> None:
        mock_observation_space = MagicMock()
        mock_observation_space.current_clear_nodes = [10, 11, 12]
        self.mock_observation_space = mock_observation_space

    def test_init_with_defaults(self):
        # Act
        agent = self._sut()

        # Assert
        self.assertIsInstance(agent,  VaccinationPolicyAgent)

    def test_available_actions(self):
        # Arrange
        agent = self._sut()

        # Assert
        self.assertListEqual(['vaccinate'], agent.available_actions)

    def test_no_actions_outside_active_period(self):
        # Arrange
        agent = self._sut(name='test_agent',
                          actions_per_turn=1,
                          start_step={'vaccinate': 25},
                          end_step={'vaccinate': 30})

        # Act
        action = agent.get_actions(self.mock_observation_space)

        # Assert
        self.assertListEqual([], list(action.keys()))

    def test_n_actions_inside_active_period(self):
        # Arrange
        agent = self._sut(name='test_agent',
                          actions_per_turn=1,
                          start_step={'vaccinate': 25},
                          end_step={'vaccinate': 30})
        agent._step = 26

        # Act
        action = agent.get_actions(self.mock_observation_space)

        # Assert
        self.assertListEqual(['vaccinate'], list(action.values()))

    def test_whole_active_period_returns_actions_with_single_actions(self):
        # Arrange
        agent = self._sut(name='test_agent',
                          actions_per_turn=1,
                          start_step={'vaccinate': 5},
                          end_step={'vaccinate': 10})

        # Act
        actions = []
        for s in range(15):
            actions.append(agent.get_actions(self.mock_observation_space))

        # Assert
        self.assertEqual(15, len(actions))
        self.assertEqual(15, agent._step)
        self.assertListEqual([0, 0, 0, 0, 0,
                              1, 1, 1, 1, 1, 1,
                              0, 0, 0, 0], [len(d.keys()) for d in actions])

    def test_whole_active_period_returns_actions_with_multiple_actions(self):
        # Arrange
        agent = self._sut(name='test_agent',
                          actions_per_turn=3,
                          start_step={'vaccinate': 5},
                          end_step={'vaccinate': 10})

        # Act
        actions = []
        for s in range(15):
            actions.append(agent.get_actions(self.mock_observation_space))

        # Assert
        self.assertEqual(15, len(actions))
        self.assertEqual(15, agent._step)
        self.assertListEqual([0, 0, 0, 0, 0,
                              3, 3, 3, 3, 3, 3,
                              0, 0, 0, 0], [len(d.keys()) for d in actions])
