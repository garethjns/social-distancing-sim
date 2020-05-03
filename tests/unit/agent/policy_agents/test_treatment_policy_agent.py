import unittest
from unittest.mock import MagicMock

from social_distancing_sim.agent.policy_agents.treatment_policy_agent import TreatmentPolicyAgent
from social_distancing_sim.environment.action_space import ActionSpace
from social_distancing_sim.environment.environment import Environment
from social_distancing_sim.environment.observation_space import ObservationSpace


class TestTreatmentPolicyAgent(unittest.TestCase):
    _sut = TreatmentPolicyAgent

    def setUp(self) -> None:
        mock_observation_space = MagicMock(spec=ObservationSpace)
        mock_observation_space.current_clear_nodes = [9, 10, 11, 12]
        mock_observation_space.current_infected_nodes = [12, 13, 14]

        mock_env = MagicMock(spec=Environment)
        mock_env.observation_space = mock_observation_space
        mock_env.action_space = ActionSpace()

        self._mock_env = mock_env

    def test_init_with_defaults(self):
        # Act
        agent = self._sut(self._mock_env)

        # Assert
        self.assertIsInstance(agent, TreatmentPolicyAgent)

    def test_available_actions(self):
        # Arrange
        agent = self._sut(self._mock_env)

        # Assert
        self.assertListEqual([4], agent.available_actions)

    def test_no_actions_outside_active_period(self):
        # Arrange
        agent = self._sut(self._mock_env,
                          name='test_agent',
                          actions_per_turn=1,
                          start_step={'treat': 25},
                          end_step={'treat': 30})

        # Act
        actions, _ = agent.get_actions()

        # Assert
        self.assertListEqual([], actions)

    def test_n_actions_inside_active_period(self):
        # Arrange
        agent = self._sut(self._mock_env,
                          name='test_agent',
                          actions_per_turn=1,
                          start_step={'treat': 25},
                          end_step={'treat': 30})
        agent._step = 26

        # Act
        actions, _ = agent.get_actions()

        # Assert
        self.assertListEqual([4], actions)

    def test_whole_active_period_returns_actions_with_single_actions(self):
        # Arrange
        agent = self._sut(self._mock_env,
                          name='test_agent',
                          actions_per_turn=1,
                          start_step={'treat': 5},
                          end_step={'treat': 10})

        # Act
        actions = []
        for _ in range(13):
            act, _ = agent.get_actions()
            actions.append(act)

        # Assert
        self.assertEqual(13, len(actions))
        self.assertEqual(13, agent._step)
        self.assertListEqual([0, 0, 0, 0, 0,
                              1, 1, 1, 1, 1, 1,
                              0, 0], [len(d) for d in actions])

    def test_whole_active_period_returns_actions_with_multiple_actions(self):
        # Arrange
        agent = self._sut(self._mock_env,
                          name='test_agent',
                          actions_per_turn=3,
                          start_step={'treat': 5},
                          end_step={'treat': 10})

        # Act
        actions = []
        for _ in range(12):
            act, _ = agent.get_actions()
            actions.append(act)

        # Assert
        self.assertEqual(12, len(actions))
        self.assertEqual(12, agent._step)
        self.assertListEqual([0, 0, 0, 0, 0,
                              3, 3, 3, 3, 3, 3,
                              0], [len(d) for d in actions])
