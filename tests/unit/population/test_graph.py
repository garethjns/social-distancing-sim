import unittest
from typing import Callable

import numpy as np

from social_distancing_sim.environment.graph import Graph


class TestGraph(unittest.TestCase):
    _sut: Callable = Graph

    def test_init_with_defaults(self):
        graph = self._sut()

        self.assertIsInstance(graph, Graph)
        self.assertAlmostEqual(10, graph.total_population, -2)

    def test_seed_consistency_when_same_specified(self):
        # Arrange
        graph1 = self._sut(seed=123)
        graph2 = self._sut(seed=123)

        # Assert
        self.assertEqual(graph1.total_population, graph2.total_population)
        self.assertListEqual(list(graph1.g_.edges), list(graph2.g_.edges))

    def test_seed_inconsistency_when_different_specified(self):
        # Arrange
        graph1 = self._sut(seed=124)
        graph2 = self._sut(seed=123)

        # Assert
        self.assertFalse(np.array((np.array(graph1.g_.edges) == np.array(graph2.g_.edges))).all())

    def test_seed_inconsistency_when_none_specified(self):
        # Arrange
        graph1 = self._sut(seed=None)
        graph2 = self._sut(seed=None)

        # Assert
        self.assertFalse(np.array((np.array(graph1.g_.edges) == np.array(graph2.g_.edges))).all())
