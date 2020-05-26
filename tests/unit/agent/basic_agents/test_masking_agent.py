import unittest
from unittest.mock import MagicMock

from social_distancing_sim.agent.basic_agents.masking_agent import MaskingAgent


class TestMaskingAgent(unittest.TestCase):
    _sut = MaskingAgent
    _mock_env = MagicMock()

    def test_init_with_defaults(self):
        # Act
        agent = self._sut(self._mock_env)

        # Assert
        self.assertIsInstance(agent, MaskingAgent)

    def test_available_actions(self):
        # Arrange
        agent = self._sut(self._mock_env)

        # Assert
        self.assertListEqual([5], agent.available_actions)
