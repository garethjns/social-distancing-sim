import unittest

from social_distancing_sim.agent.basic_agents.treatment_agent import TreatmentAgent


class TestTreatmentAgent(unittest.TestCase):
    _sut = TreatmentAgent

    def test_init_with_defaults(self):
        # Act
        agent = self._sut()

        # Assert
        self.assertIsInstance(agent, TreatmentAgent)

    def test_available_actions(self):
        # Arrange
        agent = self._sut()

        # Assert
        self.assertListEqual(['treat'], agent.available_actions)
