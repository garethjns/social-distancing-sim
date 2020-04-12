import unittest
from unittest.mock import MagicMock

from social_distancing_sim.environment.observation_space import ObservationSpace
from social_distancing_sim.environment.status import Status
import copy

class TestObservationSpace(unittest.TestCase):
    _sut: ObservationSpace = ObservationSpace
    _mock_graph = MagicMock()

    def setUp(self):
        # ObservationSpace sets status of nodes in graph to Status() on init, turn this off for these tests.
        self._sut._attach_status_to_graph = lambda x: x

    def test_init_with_defaults(self):
        obs = self._sut(graph=self._mock_graph)

        self.assertIsInstance(obs, ObservationSpace)

    def test_tested_dead_nodes_always_identified(self):
        # Arrange
        mock_graph = MagicMock()
        mock_nodes = {1: {'alive': False, 'infected': True, "immune": 0,
                          "status": Status(infected=True, last_tested=1)},
                      2: {'alive': True, 'infected': True, "immune": 0,
                          'status': Status()}}
        mock_graph.g_.nodes.data.return_value = mock_nodes.items()
        obs = self._sut(graph=mock_graph)

        # Act
        new_infections = obs.update_observed_statuses(time_step=1)

        # Assert
        self.assertEqual(0, new_infections)
        self.assertTrue(mock_nodes[1]["status"].dead)
        self.assertFalse(mock_nodes[1]["status"].alive)
        self.assertEqual(Status(), mock_nodes[2]["status"])

    def test_untested_dead_nodes_always_identified(self):
        # Arrange
        mock_graph = MagicMock()
        mock_nodes = {1: {'alive': False, 'infected': True, "immune": 0,
                          'status': Status()},
                      2: {'alive': True, 'infected': True, "immune": 0,
                          'status': Status()}}
        mock_graph.g_.nodes.data.return_value = mock_nodes.items()
        obs = self._sut(graph=mock_graph)

        # Act
        new_infections = obs.update_observed_statuses(time_step=1)

        # Assert
        self.assertEqual(0, new_infections)
        self.assertTrue(mock_nodes[1]["status"].dead)
        self.assertEqual(Status(), mock_nodes[2]["status"])

    def test_if_tested_and_immune_marked_immune(self):
        # Arrange
        mock_graph = MagicMock()
        mock_nodes = {1: {'alive': True, 'infected': False, "immune": 0.9,
                          "status": Status(last_tested=1)},
                      2: {'alive': True, 'infected': False, "immune": 0.9,
                          "status": Status()}}
        mock_graph.g_.nodes.data.return_value = mock_nodes.items()
        mock_graph.considered_immune_threshold = 0.3
        obs = self._sut(graph=mock_graph)

        # Act
        new_infections = obs.update_observed_statuses(time_step=1)

        # Assert
        self.assertEqual(0, new_infections)
        self.assertTrue(mock_nodes[1]["status"].immune)
        self.assertEqual(Status(), mock_nodes[2]["status"])

    def test_if_tested_now_clear_and_alive_marked_as_immune_or_clear(self):
        # Arrange
        mock_graph = MagicMock()
        mock_nodes = {1: {'alive': True, 'infected': False, "immune": 0.9,
                          "status": Status(infected=True, last_tested=5)},
                      2: {'alive': True, 'infected': False, "immune": 0.1,
                          "status": Status(infected=True, last_tested=5)},
                      3: {'alive': True, 'infected': False, "immune": 0,
                          "status": Status()}}
        mock_graph.g_.nodes.data.return_value = mock_nodes.items()
        mock_graph.considered_immune_threshold = 0.3
        obs = self._sut(graph=mock_graph)

        # Act
        new_infections = obs.update_observed_statuses(time_step=5)

        # Assert
        self.assertEqual(0, new_infections)
        self.assertTrue(mock_nodes[1]["status"].clear)
        self.assertTrue(mock_nodes[1]["status"].immune)
        self.assertTrue(mock_nodes[2]["status"].clear)
        self.assertFalse(mock_nodes[2]["status"].immune)
        self.assertFalse(mock_nodes[1]["status"].infected)
        self.assertFalse(mock_nodes[2]["status"].infected)
        self.assertEqual(Status(), mock_nodes[3]["status"])

    def test_if_not_tested_infected_remain_infected(self):
        # Arrange
        mock_graph = MagicMock()
        mock_nodes = {1: {'alive': True, 'infected': False, "immune": 0.9,
                          "status": Status(infected=True, last_tested=5)},
                      2: {'alive': True, 'infected': False, "immune": 0.1,
                          "status": Status(infected=True, last_tested=5)},
                      3: {'alive': True, 'infected': False, "immune": 0,
                          'status': Status()}}
        mock_graph.g_.nodes.data.return_value = mock_nodes.items()
        obs = self._sut(graph=mock_graph)

        # Act
        new_infections = obs.update_observed_statuses(time_step=10)

        # Assert
        self.assertEqual(0, new_infections)
        self.assertTrue(mock_nodes[1]["status"].infected)
        self.assertTrue(mock_nodes[2]["status"].infected)
        self.assertEqual(Status(), mock_nodes[3]["status"])

    def test_if_infected_mark_infected_only_if_tested_this_turn(self):
        # Arrange
        mock_graph = MagicMock()
        mock_nodes = {1: {'alive': True, 'infected': True, "immune": 0,
                          "status": Status(last_tested=1)},
                      2: {'alive': True, 'infected': True, "immune": 0,
                          "status": Status()}}
        mock_graph.g_.nodes.data.return_value = mock_nodes.items()
        obs = self._sut(graph=mock_graph)

        # Act
        new_infections = obs.update_observed_statuses(time_step=1)

        # Assert
        self.assertEqual(1, new_infections)
        self.assertTrue(mock_nodes[1]["status"].infected)
        self.assertEqual(Status(), mock_nodes[2]["status"])

    def test_if_infected_and_tested_test_does_not_expire(self):
        # Arrange
        mock_graph = MagicMock()
        mock_nodes = {1: {'alive': True, 'infected': True, "immune": 0,
                          "status": Status(infected=True, last_tested=1)},
                      2: {'alive': True, 'infected': True, "immune": 0,
                          "status": Status()}}
        mock_graph.g_.nodes.data.return_value = mock_nodes.items()
        obs = self._sut(graph=mock_graph)

        # Act
        new_infections = obs.update_observed_statuses(time_step=8)

        # Assert
        self.assertEqual(0, new_infections)
        self.assertTrue(mock_nodes[1]["status"].infected)
        self.assertEqual(Status(), mock_nodes[2]["status"])

    def test_if_clear_mark_clear_only_if_tested_this_turn(self):
        # Arrange
        mock_graph = MagicMock()

        mock_nodes = {1: {'alive': True, 'infected': False, "immune": 0,
                          # Was tested while infected, but on an earlier turn
                          "status": Status(infected=True, last_tested=10)},
                      2: {'alive': True, 'infected': False, "immune": 0,
                          # Was tested while infected, but on an earlier turn
                          "status": Status(clear=False, last_tested=10)},
                      3: {'alive': True, 'infected': False, "immune": 0,
                          "status": Status()}}
        mock_graph.g_.nodes.data.return_value = mock_nodes.items()
        mock_graph.considered_immune_threshold = 0.3
        obs = self._sut(graph=mock_graph)

        # Act
        new_infections = obs.update_observed_statuses(time_step=10)

        # Assert
        self.assertEqual(0, new_infections)
        self.assertTrue(mock_nodes[1]["status"].clear)
        self.assertTrue(mock_nodes[2]["status"].clear)
        self.assertFalse(mock_nodes[1]["status"].infected)
        self.assertFalse(mock_nodes[2]["status"].infected)
        self.assertEqual(Status(), mock_nodes[3]["status"])

    def test_clear_and_immune_tests_expire_after_validity_period(self):
        # Arrange
        mock_graph = MagicMock()
        mock_nodes = {1: {'alive': True, 'infected': False, "immune": 0,
                          "status": Status(clear=True, last_tested=1)},
                      2: {'alive': True, 'infected': True, "immune": 0,
                          "status": Status(infected=True, last_tested=1)},
                      3: {'alive': True, 'infected': False, "immune": 0.5,
                          "status": Status(immune=True, last_tested=1)},
                      4: {'alive': True, 'infected': False, "immune": 0,
                          "status": Status()}}
        mock_graph.g_.nodes.data.return_value = mock_nodes.items()
        obs = self._sut(graph=mock_graph)

        # Act
        new_infections = obs.update_observed_statuses(time_step=8)

        # Assert
        self.assertEqual(0, new_infections)
        self.assertIsNone(mock_nodes[1]["status"].clear)
        self.assertTrue(mock_nodes[2]["status"].infected)
        self.assertIsNone(mock_nodes[3]["status"].immune)
        self.assertEqual(Status(), mock_nodes[4]["status"])
