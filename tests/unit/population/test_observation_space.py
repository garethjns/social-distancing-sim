import unittest
from unittest.mock import MagicMock

from social_distancing_sim.population.observation_space import ObservationSpace


class TestObservationSpace(unittest.TestCase):
    _sut = ObservationSpace
    _mock_graph = MagicMock()

    def test_init_with_defaults(self):
        obs = self._sut(graph=self._mock_graph)

        self.assertIsInstance(obs, ObservationSpace)
