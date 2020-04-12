import unittest

from social_distancing_sim.agent.isolation_agent import IsolationAgent


class TestVaccinationAgent(unittest.TestCase):
    _sut = IsolationAgent

    def test_init_with_defaults(self):
        # Act
        agent = self._sut()

        # Assert
        self.assertIsInstance(agent, IsolationAgent)

    def test_available_actions(self):
        # Arrange
        agent = self._sut()

        # Assert
        self.assertListEqual(['isolate', 'reconnect'], agent.available_actions)
