import unittest
from unittest.mock import MagicMock

from social_distancing_sim.agent.basic_agents.random_agent import RandomAgent


class TestRandomAgent(unittest.TestCase):
    _sut = RandomAgent
    _mock_env = MagicMock()

    def test_init_with_defaults(self):
        # Act
        agent = self._sut(self._mock_env)

        # Assert
        self.assertIsInstance(agent, RandomAgent)

    def test_available_actions(self):
        # Arrange
        agent = self._sut(self._mock_env)

        # Assert
        self.assertListEqual([1, 2], agent.available_actions)
