import unittest
from unittest.mock import MagicMock

from social_distancing_sim.agent.basic_agents.isolation_agent import IsolationAgent


class TestVaccinationAgent(unittest.TestCase):
    _sut = IsolationAgent
    _mock_env = MagicMock()

    def test_init_with_defaults(self):
        # Act
        agent = self._sut(self._mock_env)

        # Assert
        self.assertIsInstance(agent, IsolationAgent)

    def test_available_actions(self):
        # Arrange
        agent = self._sut(self._mock_env)

        # Assert
        self.assertListEqual([2, 3], agent.available_actions)
