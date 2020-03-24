import unittest

from social_distancing_sim.population.history import History


class TestHistory(unittest.TestCase):
    _sut = History

    def test_init_with_defaults(self):
        # Act
        history = self._sut.with_defaults()

        # Assert
        self.assertIsInstance(history, History)
        self.assertIn("Current infections", list(history.keys()))
