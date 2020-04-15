import unittest
from unittest.mock import patch
from social_distancing_sim.agent.agent_base import AgentBase
from social_distancing_sim.environment.observation_space import ObservationSpace


class TestAgentBase(unittest.TestCase):

    @patch.multiple(AgentBase, __abstractmethods__=set())
    def test_sample_action(self):
        # Arrange
        mock_obs = unittest.mock.MagicMock(kind=ObservationSpace)
        mock_obs.current_alive_nodes = [1, 2, 3]
        agent = AgentBase(actions_per_turn=1)

        # Act
        actions = agent.sample(obs=mock_obs)

        # Assert
        self.assertIsInstance(actions, dict)
        self.assertEqual(1, len(actions.keys()))

    @patch.multiple(AgentBase, __abstractmethods__=set())
    def test_limit_actions_by_available_targets(self):
        # Arrange
        mock_obs = unittest.mock.MagicMock(kind=ObservationSpace)
        mock_obs.current_alive_nodes = [1, 2, 3]
        agent = AgentBase(actions_per_turn=6)

        # Act
        actions = agent.sample(obs=mock_obs)

        # Assert
        self.assertIsInstance(actions, dict)
        self.assertEqual(3, len(actions.keys()))
