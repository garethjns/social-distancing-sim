import unittest

from social_distancing_sim.agent.vaccination_agent import VaccinationAgent


class TestVaccinationAgent(unittest.TestCase):
    _sut = VaccinationAgent

    def test_init_with_defaults(self):
        # Act
        agent = self._sut()

        # Assert
        self.assertIsInstance(agent, VaccinationAgent)

    def test_available_actions(self):
        # Arrange
        agent = self._sut()

        # Assert
        self.assertListEqual(['vaccinate'], agent.available_actions)
