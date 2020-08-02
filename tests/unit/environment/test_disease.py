import copy
import unittest
import warnings
from typing import Callable
from unittest.mock import MagicMock

import numpy as np

from social_distancing_sim.environment.disease import Disease


class TestDisease(unittest.TestCase):
    _sut: Callable = Disease

    def setUp(self):
        self._mock_node = {}
        self._mock_infected_node = {"infected": 5}
        self._mock_immune_node = {"immune": True}
        self._mock_graph = [copy.deepcopy(self._mock_node) for _ in range(20000)]
        self._mock_immune_graph = [copy.deepcopy(self._mock_immune_node) for _ in range(5000)]

    def test_init_with_defaults(self):
        # Act
        disease = self._sut()

        # Assert
        self.assertIsInstance(disease, Disease)

    def test_modified_virulence_unchanged_with_no_immunity(self):
        # Arrange
        disease = self._sut(virulence=0.005)

        # Act
        mv = disease.modified_virulence(0)

        self.assertAlmostEqual(0.005, mv)

    def test_modified_virulence_changed_with_immunity(self):
        # Arrange
        disease = self._sut(virulence=0.005)

        # Act
        mv = disease.modified_virulence(0.5)

        self.assertAlmostEqual(0.0025, mv)

    def test_modified_virulence_min_cap(self):
        # Arrange
        disease = self._sut(virulence=0.005)

        # Act
        mv = disease.modified_virulence(10000)

        self.assertAlmostEqual(1e-7, mv)

    def test_modified_negative_immunity(self):
        # Arrange
        disease = self._sut(virulence=0.005)

        # Act
        mv = disease.modified_virulence(-0.5)

        self.assertAlmostEqual(0.0075, mv)

    def test_modified_virulence_max_cap(self):
        # Arrange
        disease = self._sut(virulence=0.005)

        # Act
        mv = disease.modified_virulence(-1000)

        self.assertAlmostEqual(0.999, mv)

    def test_modified_virulence_with_multiple_05_modifiers(self):
        # Arrange
        disease = self._sut(virulence=1)

        # Act
        mv = disease.modified_virulence(modifiers=[0.5, 0.5, 0.5])

        self.assertAlmostEqual(0.125, mv)  # 87.5% protection

    def test_modified_virulence_with_multiple_low_modifiers(self):
        # Arrange
        disease = self._sut(virulence=1)

        # Act
        mv1 = disease.modified_virulence(modifiers=[0.1, 0.1, 0.1])  # 27.7% protection
        mv2 = disease.modified_virulence(modifiers=[0.1, 0.1, 0.2])  # 36% protection

        self.assertAlmostEqual(0.729, mv1)
        self.assertGreater(mv1, mv2)

    def test_modified_virulence_with_multiple_high_modifiers(self):
        # Arrange
        disease = self._sut(virulence=1)

        # Act
        mv1 = disease.modified_virulence(modifiers=[0.9, 0.9, 0.9])
        mv2 = disease.modified_virulence(modifiers=[0.9, 0.8, 0.9])

        self.assertAlmostEqual(0.0009, mv1, 3)
        self.assertLess(mv1, mv2)

    def test_modified_virulence_with_single_low_modifier(self):
        # Arrange
        disease = self._sut(virulence=1)

        # Act
        mv1 = disease.modified_virulence(modifiers=[0.5, 0.5, 0.9])
        mv2 = disease.modified_virulence(modifiers=[0.5, 0.9, 0.5])

        self.assertAlmostEqual(0.025, mv1)
        self.assertAlmostEqual(mv1, mv2)

    def test_modified_virulence_with_single_high_modifier(self):
        # Arrange
        disease = self._sut(virulence=1)

        # Act
        mv1 = disease.modified_virulence(modifiers=[0.9, 0.5, 0.5])
        mv2 = disease.modified_virulence(modifiers=[0.5, 0.5, 0.9])

        self.assertAlmostEqual(0.025, mv1)
        self.assertAlmostEqual(mv1, mv2)

    def test_force_infect(self):
        # Arrange
        disease = self._sut()

        # Act
        infected_node = disease.force_infect(self._mock_node)

        self.assertIn("infected", infected_node.keys())
        self.assertEqual(infected_node["infected"], 1)

    def test_seed_consistency_when_same_specified(self):
        # Arrange
        disease1 = self._sut(seed=123)
        disease2 = self._sut(seed=123)

        # Act
        infections1 = [disease1.try_to_infect(self._mock_infected_node, node)
                       for node in copy.deepcopy(self._mock_graph)]
        infections2 = [disease2.try_to_infect(self._mock_infected_node, node)
                       for node in copy.deepcopy(self._mock_graph)]

        # Assert
        self.assertIn(0, [n.get("infected", 0) for n in infections1])
        self.assertIn(1, [n.get("infected", 0) for n in infections1])
        self.assertListEqual(infections1, infections2)

    def test_seed_inconsistency_when_different_specified(self):
        # Arrange
        disease1 = self._sut(seed=124)
        disease2 = self._sut(seed=123)

        # Act
        infections1 = [disease1.try_to_infect(self._mock_infected_node, node)
                       for node in copy.deepcopy(self._mock_graph)]
        infections2 = [disease2.try_to_infect(self._mock_infected_node, node)
                       for node in copy.deepcopy(self._mock_graph)]


        # Assert
        self.assertIn(0, [n.get("infected", 0) for n in infections1])
        self.assertIn(1, [n.get("infected", 0) for n in infections1])
        self.assertFalse((np.array(infections1) == np.array(infections2)).all())

    def test_seed_inconsistency_when_none_specified(self):
        # Arrange
        disease1 = self._sut(seed=None)
        disease2 = self._sut(seed=None)

        # Act
        infections1 = [disease1.try_to_infect(self._mock_infected_node, node)
                       for node in copy.deepcopy(self._mock_graph)]
        infections2 = [disease2.try_to_infect(self._mock_infected_node, node)
                       for node in copy.deepcopy(self._mock_graph)]

        # Assert
        self.assertIn(0, [n["infected"] for n in infections1])
        self.assertIn(1, [n["infected"] for n in infections1])
        self.assertFalse((np.array(infections1) == np.array(infections2)).all())

    def test_seed_consistency_when_same_specified_multiple(self):
        # Arrange
        disease1 = self._sut(seed=123, virulence=0.5)
        disease2 = self._sut(seed=123, virulence=0.5)
        graph1 = copy.deepcopy(self._mock_graph)
        graph1.append({'infected': 5})
        graph2 = copy.deepcopy(graph1)

        # Act
        infections1 = disease1.try_to_infect_multiple(source_node=graph1[-1], target_nodes=graph1[0:200])
        infections2 = disease2.try_to_infect_multiple(source_node=graph2[-1], target_nodes=graph2[0:200])

        # Assert
        self.assertIn(0, infections1)
        self.assertIn(1, infections1)
        self.assertListEqual(infections1, infections2)

    def test_seed_inconsistency_when_different_specified_multiple(self):
        # Arrange
        disease1 = self._sut(seed=124, virulence=0.5)
        disease2 = self._sut(seed=123, virulence=0.5)
        graph1 = copy.deepcopy(self._mock_graph)
        graph1.append({'infected': 5})
        graph2 = copy.deepcopy(graph1)

        # Act
        infections1 = disease1.try_to_infect_multiple(source_node=graph1[-1], target_nodes=graph1[0:200])
        infections2 = disease2.try_to_infect_multiple(source_node=graph2[-1], target_nodes=graph2[0:200])

        # Assert
        self.assertIn(0, infections1)
        self.assertIn(1, infections1)
        self.assertFalse((np.array(infections1) == np.array(infections2)).all())

    def test_seed_consistency_when_none_specified_multiple(self):
        # Arrange
        disease1 = self._sut(seed=None, virulence=0.5)
        disease2 = self._sut(seed=None, virulence=0.5)
        graph1 = copy.deepcopy(self._mock_graph)
        graph1.append({'infected': 5})
        graph2 = copy.deepcopy(graph1)

        # Act
        infections1 = disease1.try_to_infect_multiple(source_node=graph1[-1], target_nodes=graph1[0:200])
        infections2 = disease2.try_to_infect_multiple(source_node=graph2[-1], target_nodes=graph2[0:200])

        # Assert
        self.assertIn(0, infections1)
        self.assertIn(1, infections1)
        self.assertFalse((np.array(infections1) == np.array(infections2)).all())

    def test_cannot_infect_immune_node(self):
        # Arrange
        disease = self._sut()

        # Act
        infections = [disease.try_to_infect(node, node) for node in copy.deepcopy(self._mock_immune_graph)]

        self.assertListEqual([n.get("infected", 0) for n in infections], [0] * len(infections))

    def test_try_to_infect_with_infectious_source(self):
        disease = self._sut(virulence=1)
        source_node = {"infected": 1}
        target_node = {}

        target_node = disease.try_to_infect(source_node=source_node, target_node=target_node)

        self.assertGreater(target_node["infected"], 0)

    def test_try_to_infect_with_non_infectious_source(self):
        disease = self._sut()
        source_node = {"infected": 0}
        target_node = {}

        for _ in range(10000):
            target_node = disease.try_to_infect(source_node=source_node, target_node=target_node)

            self.assertEqual(0, target_node.get("infected", 0))

    def test_try_to_infect_multiple_with_infectious_source(self):
        disease = self._sut(virulence=1)
        source_node = {"infected": 1}
        target_nodes = [{"infected": 0}, {"infected": 0}, {"infected": 0}]

        new_infections = disease.try_to_infect_multiple(source_node=source_node, target_nodes=target_nodes)

        self.assertListEqual([1, 1, 1], new_infections)

    def test_try_to_infect_multiple_with_non_infectious_source(self):
        disease = self._sut(virulence=1)
        source_node = {"infected": 0}
        target_nodes = [{"infected": 0}] * 20000

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            new_infections = disease.try_to_infect_multiple(source_node=source_node, target_nodes=target_nodes)

        self.assertEqual(0, np.sum(new_infections))
        self.assertEqual('Attempted to infect with non-infectious source, returning no new infections.',
                         str(w[0].message))

    def test_give_immunity_gives_immunity_to_node(self):
        # Arrange
        nodes = {0: {'status': MagicMock()}}

        # Act
        node = self._sut().give_immunity(nodes[0])

        # Assert
        self.assertGreater(node['immune'], 0)

    def test_give_decay_immunity_lowers_nodes_immunity(self):
        # Arrange
        nodes = {0: {'status': MagicMock(),
                     'immune': 0.5}}

        # Act
        node = self._sut().decay_immunity(nodes[0])

        # Assert
        self.assertLess(node['immune'], 0.5)
