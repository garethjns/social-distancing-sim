import copy
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

    def test_mask_nodes_adds_mask_to_target_node(self):
        # Arrange
        g = self._sut()

        # Act
        g.mask_node(0)

        # Assert
        self.assertGreater(g.g_.nodes[0]['mask'], 0)

    def test_unmask_node_removes_mask_from_target_node(self):
        # Arrange
        g = self._sut()
        g.g_.nodes[0]["mask"] = 0.6

        # Act
        g.unmask_node(0)

        # Assert
        self.assertEqual(g.g_.nodes[0]['mask'], 0)

    def test_isolate_node_removes_and_saves_connections(self):
        # Arrange
        g = self._sut()
        old_connections = copy.deepcopy(list(g.g_.edges(0)))

        # Act
        g.isolate_node(0, effectiveness=1)

        # Assert
        self.assertListEqual([], list(g.g_.edges(0)))
        self.assertListEqual(old_connections, g.g_.nodes[0]["_edges"])

    def test_reconnect_node_restores_saved_connections(self):
        # Arrange
        g = self._sut()
        current_connections = copy.deepcopy(list(g.g_.edges(0)))
        previously_removed_connections = [(0, 1), (0, 2), (0, 3)]
        g.g_.nodes[0]["_edges"] = copy.deepcopy(previously_removed_connections)

        # Act
        g.reconnect_node(0, effectiveness=1)

        # Assert
        for edge in previously_removed_connections:
            self.assertIn(edge, list(g.g_.edges(0)))
        for edge in current_connections:
            self.assertIn(edge, list(g.g_.edges(0)))
        self.assertListEqual([], list(g.g_.nodes[0]["_edges"]))
