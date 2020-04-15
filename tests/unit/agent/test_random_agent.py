import unittest

from social_distancing_sim.agent.random_agent import RandomAgent


class TestRandomAgent(unittest.TestCase):
    _sut = RandomAgent

    def test_init_with_defaults(self):
        # Act
        agent = self._sut()

        # Assert
        self.assertIsInstance(agent, RandomAgent)

    def test_available_actions(self):
        # Arrange
        agent = self._sut()

        # Assert
        self.assertListEqual(['vaccinate', 'isolate'], agent.available_actions)
