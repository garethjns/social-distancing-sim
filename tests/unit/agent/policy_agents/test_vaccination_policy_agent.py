import unittest
from unittest.mock import MagicMock

from social_distancing_sim.agent.policy_agents.vaccination_policy_agent import VaccinationPolicyAgent
from social_distancing_sim.environment.action_space import ActionSpace
from social_distancing_sim.environment.environment import Environment
from social_distancing_sim.environment.observation_space import ObservationSpace


class TestVaccinationPolicyAgent(unittest.TestCase):
    _sut = VaccinationPolicyAgent

    def setUp(self) -> None:

        mock_observation_space = MagicMock(spec=ObservationSpace)
        mock_observation_space.current_clear_nodes = [9, 10, 11, 12]

        mock_env = MagicMock(spec=Environment)
        mock_env.observation_space = mock_observation_space
        mock_env.action_space = ActionSpace()

        self._mock_env = mock_env

    def test_init_with_defaults(self):
        # Act
        agent = self._sut(self._mock_env)

        # Assert
        self.assertIsInstance(agent, VaccinationPolicyAgent)

    def test_available_actions(self):
        # Arrange
        agent = self._sut(self._mock_env)

        # Assert
        self.assertListEqual([1], agent.available_actions)

    def test_no_actions_outside_active_period(self):
        # Arrange
        agent = self._sut(self._mock_env,
                          name='test_agent',
                          actions_per_turn=1,
                          start_step={'vaccinate': 25},
                          end_step={'vaccinate': 30})

        # Act
        actions, _ = agent.get_actions()

        # Assert
        self.assertListEqual([], actions)

    def test_n_actions_inside_active_period(self):
        # Arrange
        agent = self._sut(self._mock_env,
                          name='test_agent',
                          actions_per_turn=1,
                          start_step={'vaccinate': 25},
                          end_step={'vaccinate': 30})
        agent._step = 26

        # Act
        actions, _ = agent.get_actions()

        # Assert
        self.assertListEqual([1], actions)

    def test_whole_active_period_returns_actions_with_single_actions(self):
        # Arrange
        agent = self._sut(self._mock_env,
                          name='test_agent',
                          actions_per_turn=1,
                          start_step={'vaccinate': 5},
                          end_step={'vaccinate': 10})

        # Act
        actions = []
        for _ in range(15):
            act, _ = agent.get_actions()
            actions.append(act)

        # Assert
        self.assertEqual(15, len(actions))
        self.assertEqual(15, agent._step)
        self.assertListEqual([0, 0, 0, 0, 0,
                              1, 1, 1, 1, 1, 1,
                              0, 0, 0, 0], [len(d) for d in actions])

    def test_whole_active_period_returns_actions_with_multiple_actions(self):
        # Arrange
        agent = self._sut(self._mock_env,
                          name='test_agent',
                          actions_per_turn=3,
                          start_step={'vaccinate': 5},
                          end_step={'vaccinate': 10})

        # Act
        actions = []
        for _ in range(14):
            act, _ = agent.get_actions()
            actions.append(act)

        # Assert
        self.assertEqual(14, len(actions))
        self.assertEqual(14, agent._step)
        self.assertListEqual([0, 0, 0, 0, 0,
                              3, 3, 3, 3, 3, 3,
                              0, 0, 0], [len(d) for d in actions])
