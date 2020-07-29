import unittest
from unittest.mock import MagicMock

from social_distancing_sim.agent.basic_agents.treatment_agent import TreatmentAgent


class TestTreatmentAgent(unittest.TestCase):
    _sut = TreatmentAgent
    _mock_env = MagicMock()

    def test_init_with_defaults(self):
        # Act
        agent = self._sut(self._mock_env)

        # Assert
        self.assertIsInstance(agent, TreatmentAgent)

    def test_available_actions(self):
        # Arrange
        agent = self._sut(self._mock_env)

        # Assert
        self.assertListEqual([4], agent.available_actions)
