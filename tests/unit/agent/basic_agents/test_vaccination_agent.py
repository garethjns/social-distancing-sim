import unittest
from unittest.mock import MagicMock

from social_distancing_sim.agent.basic_agents.vaccination_agent import VaccinationAgent


class TestVaccinationAgent(unittest.TestCase):
    _sut = VaccinationAgent
    _mock_env = MagicMock()

    def test_init_with_defaults(self):
        # Act
        agent = self._sut(self._mock_env)

        # Assert
        self.assertIsInstance(agent, VaccinationAgent)

    def test_available_actions(self):
        # Arrange
        agent = self._sut(self._mock_env)

        # Assert
        self.assertListEqual([1], agent.available_actions)
