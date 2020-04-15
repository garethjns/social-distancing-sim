import copy
import unittest
from typing import Callable

import numpy as np

from social_distancing_sim.environment.disease import Disease


class TestDisease(unittest.TestCase):
    _sut: Callable = Disease

    def setUp(self):
        self._mock_node = {}
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
        infections1 = [disease1.try_to_infect(node) for node in copy.deepcopy(self._mock_graph)]
        infections2 = [disease2.try_to_infect(node) for node in copy.deepcopy(self._mock_graph)]

        # Assert
        self.assertIn(0, [n["infected"] for n in infections1])
        self.assertIn(1, [n["infected"] for n in infections1])
        self.assertListEqual(infections1, infections2)

    def test_seed_inconsistency_when_different_specified(self):
        # Arrange
        disease1 = self._sut(seed=124)
        disease2 = self._sut(seed=123)

        # Act
        infections1 = [disease1.try_to_infect(node) for node in copy.deepcopy(self._mock_graph)]
        infections2 = [disease2.try_to_infect(node) for node in copy.deepcopy(self._mock_graph)]

        # Assert
        self.assertIn(0, [n["infected"] for n in infections1])
        self.assertIn(1, [n["infected"] for n in infections1])
        self.assertFalse((np.array(infections1) == np.array(infections2)).all())

    def test_seed_inconsistency_when_none_specified(self):
        # Arrange
        disease1 = self._sut(seed=None)
        disease2 = self._sut(seed=None)

        # Act
        infections1 = [disease1.try_to_infect(node) for node in copy.deepcopy(self._mock_graph)]
        infections2 = [disease2.try_to_infect(node) for node in copy.deepcopy(self._mock_graph)]

        # Assert
        self.assertIn(0, [n["infected"] for n in infections1])
        self.assertIn(1, [n["infected"] for n in infections1])
        self.assertFalse((np.array(infections1) == np.array(infections2)).all())

    def test_cannot_infect_immune_node(self):
        # Arrange
        disease = self._sut()

        # Act
        infections = [disease.try_to_infect(node) for node in copy.deepcopy(self._mock_immune_graph)]

        self.assertListEqual([n["infected"] for n in infections], [0] * len(infections))
