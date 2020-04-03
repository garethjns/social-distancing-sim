import unittest
from unittest.mock import MagicMock

from social_distancing_sim.population.observation_space import ObservationSpace


class TestObservationSpace(unittest.TestCase):
    _sut: ObservationSpace = ObservationSpace
    _mock_graph = MagicMock()

    def test_init_with_defaults(self):
        obs = self._sut(graph=self._mock_graph)

        self.assertIsInstance(obs, ObservationSpace)

    def test_tested_dead_nodes_always_identified(self):
        # Arrange
        mock_graph = MagicMock()
        mock_nodes = {1: {'alive': False, 'infected': True, "immune": False,
                          "status": "infected", "last_tested": 1},
                      2: {'alive': True, 'infected': True, "immune": False}}
        mock_graph.g_.nodes.data.return_value = mock_nodes.items()
        obs = self._sut(graph=mock_graph)

        # Act
        new_infections = obs.update_observed_statuses(time_step=1)

        # Assert
        self.assertEqual(0, new_infections)
        self.assertEqual('dead', mock_nodes[1]["status"])
        self.assertEqual('', mock_nodes[2].get("status", ''))

    def test_untested_dead_nodes_always_identified(self):
        # Arrange
        mock_graph = MagicMock()
        mock_nodes = {1: {'alive': False, 'infected': True, "immune": False},
                      2: {'alive': True, 'infected': True, "immune": False}}
        mock_graph.g_.nodes.data.return_value = mock_nodes.items()
        obs = self._sut(graph=mock_graph)

        # Act
        new_infections = obs.update_observed_statuses(time_step=1)

        # Assert
        self.assertEqual(0, new_infections)
        self.assertEqual('dead', mock_nodes[1]["status"])
        self.assertEqual('', mock_nodes[2].get("status", ''))

    def test_if_known_immune_always_immune(self):
        # Arrange
        mock_graph = MagicMock()
        mock_nodes = {1: {'alive': True, 'infected': False, "immune": True,
                          "status": "immune", "last_tested": 1},
                      2: {'alive': True, 'infected': False, "immune": True}}
        mock_graph.g_.nodes.data.return_value = mock_nodes.items()
        obs = self._sut(graph=mock_graph)

        # Act
        new_infections = obs.update_observed_statuses(time_step=100)

        # Assert
        self.assertEqual(0, new_infections)
        self.assertEqual("immune", mock_nodes[1]["status"])
        self.assertEqual('', mock_nodes[2].get("status", ''))

    def test_if_known_was_infected_now_clear_and_alive_marked_as_immune(self):
        # Arrange
        mock_graph = MagicMock()
        mock_nodes = {1: {'alive': True, 'infected': False, "immune": True,
                          "status": "infected", "last_tested": 5},
                      2: {'alive': True, 'infected': False, "immune": True}}
        mock_graph.g_.nodes.data.return_value = mock_nodes.items()
        obs = self._sut(graph=mock_graph)

        # Act
        new_infections = obs.update_observed_statuses(time_step=10)

        # Assert
        self.assertEqual(0, new_infections)
        self.assertEqual("immune", mock_nodes[1]["status"])
        self.assertEqual('', mock_nodes[2].get("status", ''))

    def test_if_infected_mark_infected_only_if_tested_this_turn(self):
        # Arrange
        mock_graph = MagicMock()
        mock_nodes = {1: {'alive': True, 'infected': True, "immune": False,
                          "last_tested": 1},
                      2: {'alive': True, 'infected': True, "immune": False}}
        mock_graph.g_.nodes.data.return_value = mock_nodes.items()
        obs = self._sut(graph=mock_graph)

        # Act
        new_infections = obs.update_observed_statuses(time_step=1)

        # Assert
        self.assertEqual(1, new_infections)
        self.assertEqual("infected", mock_nodes[1]["status"])
        self.assertEqual('', mock_nodes[2].get("status", ''))

    def test_if_infected_and_tested_test_does_not_expire(self):
        # Arrange
        mock_graph = MagicMock()
        mock_nodes = {1: {'alive': True, 'infected': True, "immune": False,
                          "status": "infected", "last_tested": 1},
                      2: {'alive': True, 'infected': True, "immune": False}}
        mock_graph.g_.nodes.data.return_value = mock_nodes.items()
        obs = self._sut(graph=mock_graph)

        # Act
        new_infections = obs.update_observed_statuses(time_step=8)

        # Assert
        self.assertEqual(0, new_infections)
        self.assertEqual("infected", mock_nodes[1]["status"])
        self.assertEqual('', mock_nodes[2].get("status", ''))

    def test_if_clear_mark_clear_only_if_tested_this_turn(self):
        # Arrange
        mock_graph = MagicMock()
        mock_nodes = {1: {'alive': True, 'infected': False, "immune": False,
                          "status": "infected", "last_tested": 10},  # Was tested while infected, but on an earlier turn
                      2: {'alive': True, 'infected': False, "immune": False}}
        mock_graph.g_.nodes.data.return_value = mock_nodes.items()
        obs = self._sut(graph=mock_graph)

        # Act
        new_infections = obs.update_observed_statuses(time_step=10)

        # Assert
        self.assertEqual(0, new_infections)
        self.assertEqual("clear", mock_nodes[1]["status"])
        self.assertEqual('', mock_nodes[2].get("status", ''))

    def test_clear_tests_expire_after_validity_period(self):
        # Arrange
        mock_graph = MagicMock()
        mock_nodes = {1: {'alive': True, 'infected': False, "immune": False,
                          "status": "clear", "last_tested": 1},
                      2: {'alive': True, 'infected': True, "immune": False,
                          "status": "infected", "last_tested": 1},
                      3: {'alive': True, 'infected': False, "immune": False,
                          "status": "immune", "last_tested": 1},
                      4: {'alive': True, 'infected': False, "immune": False}}
        mock_graph.g_.nodes.data.return_value = mock_nodes.items()
        obs = self._sut(graph=mock_graph)

        # Act
        new_infections = obs.update_observed_statuses(time_step=8)

        # Assert
        self.assertEqual(0, new_infections)
        self.assertEqual("", mock_nodes[1]["status"])
        self.assertEqual("infected", mock_nodes[2]["status"])
        self.assertEqual("immune", mock_nodes[3]["status"])
        self.assertEqual('', mock_nodes[4].get("status", ''))
