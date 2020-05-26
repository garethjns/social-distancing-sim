import copy
import unittest

import social_distancing_sim.environment as env
from social_distancing_sim.environment.action_space import ActionSpace


class TestActionSpace(unittest.TestCase):
    _sut = ActionSpace

    def setUp(self) -> None:
        self.env = env.Environment(name='Test env',
                                   disease=env.Disease(name='test disease'),
                                   observation_space=env.ObservationSpace(graph=env.Graph(community_n=30,
                                                                                          community_p_in=1,
                                                                                          community_p_out=0.9,
                                                                                          community_size_mean=5,
                                                                                          seed=123)))

    def test_expected_default_actions_are_available(self):
        # Arrange
        action_space = self._sut()

        # Act
        available_actions = action_space.available_actions

        # Assert
        self.assertListEqual([0, 1, 2, 3, 4, 5], available_actions)

    def test_vaccinate_action_adds_default_immunity(self):
        # Arrange
        action_space = self._sut()

        # Act
        cost = action_space.vaccinate(env=self.env, target_node_id=1, step=1)

        # Assert
        self.assertEqual(action_space.vaccinate_cost, cost)
        self.assertFalse(self.env.observation_space.graph.g_.nodes[0]["immune"] > 0)
        self.assertTrue(self.env.observation_space.graph.g_.nodes[1]["immune"] > 0)
        self.assertFalse(self.env.observation_space.graph.g_.nodes[0]["status"].immune)
        self.assertTrue(self.env.observation_space.graph.g_.nodes[1]["status"].immune)

    def test_vaccinate_action_adds_extra_immunity_if_specified(self):
        # Arrange
        action_space = self._sut(vaccinate_efficiency=1)

        # Act
        cost = action_space.vaccinate(env=self.env, target_node_id=1, step=1)

        # Assert
        self.assertEqual(action_space.vaccinate_cost, cost)
        self.assertFalse(self.env.observation_space.graph.g_.nodes[0]["immune"] > 0)
        self.assertTrue(self.env.observation_space.graph.g_.nodes[1]["immune"] == 1)
        self.assertFalse(self.env.observation_space.graph.g_.nodes[0]["status"].immune)
        self.assertTrue(self.env.observation_space.graph.g_.nodes[1]["status"].immune)

    def test_isolate_action_isolates_all_with_full_effectiveness(self):
        # Arrange
        action_space = self._sut(isolate_efficiency=1)

        # Act
        cost = action_space.isolate(env=self.env, target_node_id=1, step=1)

        # Assert
        self.assertEqual(action_space.isolate_cost, cost)
        self.assertFalse(self.env.observation_space.graph.g_.nodes[0]["isolated"])
        self.assertTrue(self.env.observation_space.graph.g_.nodes[1]["isolated"])
        self.assertTrue(len(self.env.observation_space.graph.g_.edges(0)) > 0)
        self.assertEqual(0, len(self.env.observation_space.graph.g_.edges(1)))
        self.assertEqual(0, len(self.env.observation_space.graph.g_.nodes[0]["_edges"]))
        self.assertTrue(len(self.env.observation_space.graph.g_.nodes[1]["_edges"]) > 0)

    def test_isolate_action_isolates_does_not_isolate_all_with_limited_effectiveness(self):
        # Arrange
        action_space = self._sut(isolate_efficiency=0.1)

        # Act
        cost = action_space.isolate(env=self.env, target_node_id=1, step=1)

        # Assert
        self.assertEqual(action_space.isolate_cost, cost)
        self.assertFalse(self.env.observation_space.graph.g_.nodes[0]["isolated"])
        self.assertTrue(self.env.observation_space.graph.g_.nodes[1]["isolated"])
        self.assertTrue(len(self.env.observation_space.graph.g_.edges(0)) > 0)
        self.assertTrue(len(self.env.observation_space.graph.g_.edges(1)) > 0)
        self.assertEqual(0, len(self.env.observation_space.graph.g_.nodes[0]["_edges"]))
        self.assertTrue(len(self.env.observation_space.graph.g_.nodes[1]["_edges"]) > 0)

    def test_reconnect_action_reconnects_all_with_full_effectiveness(self):
        # Arrange
        action_space = self._sut(isolate_efficiency=1,
                                 reconnect_efficiency=1)
        _ = action_space.isolate(env=self.env, target_node_id=1, step=1)

        # Act
        cost = action_space.reconnect(env=self.env, target_node_id=1, step=1)

        # Assert
        self.assertEqual(action_space.reconnect_cost, cost)
        self.assertFalse(self.env.observation_space.graph.g_.nodes[0]["isolated"])
        self.assertFalse(self.env.observation_space.graph.g_.nodes[1]["isolated"])
        self.assertTrue(len(self.env.observation_space.graph.g_.edges(0)) > 0)
        self.assertTrue(len(self.env.observation_space.graph.g_.edges(1)) > 0)
        self.assertEqual(0, len(self.env.observation_space.graph.g_.nodes[0]["_edges"]))
        self.assertEqual(0, len(self.env.observation_space.graph.g_.nodes[1]["_edges"]))

    def test_reconnect_action_does_not_reconnect_all_with_limited_effectiveness(self):
        # Arrange
        action_space = self._sut(isolate_efficiency=1,
                                 reconnect_efficiency=0.2)
        _ = action_space.isolate(env=self.env, target_node_id=1, step=1)

        # Act
        cost = action_space.reconnect(env=self.env, target_node_id=1, step=1)

        # Assert
        self.assertEqual(action_space.reconnect_cost, cost)
        self.assertFalse(self.env.observation_space.graph.g_.nodes[0]["isolated"])
        self.assertTrue(self.env.observation_space.graph.g_.nodes[1]["isolated"])
        self.assertTrue(len(self.env.observation_space.graph.g_.edges(0)) > 0)
        self.assertTrue(len(self.env.observation_space.graph.g_.edges(1)) > 0)
        self.assertEqual(0, len(self.env.observation_space.graph.g_.nodes[0]["_edges"]))
        self.assertTrue(len(self.env.observation_space.graph.g_.nodes[1]["_edges"]) > 0)

    def test_reconnect_action_eventually_reconnects_all_with_limited_effectiveness_and_repeated_calls(self):
        # Arrange
        action_space = self._sut(isolate_efficiency=1,
                                 reconnect_efficiency=0.7)
        original_edges = copy.deepcopy(list(self.env.observation_space.graph.g_.edges(1)))
        _ = action_space.isolate(env=self.env, target_node_id=1, step=1)

        # Act
        for _ in range(10):
            cost = action_space.reconnect(env=self.env, target_node_id=1, step=1)

        # Assert
        self.assertEqual(action_space.reconnect_cost, cost)
        self.assertFalse(self.env.observation_space.graph.g_.nodes[0]["isolated"])
        self.assertFalse(self.env.observation_space.graph.g_.nodes[1]["isolated"])
        self.assertTrue(len(self.env.observation_space.graph.g_.edges(0)) > 0)
        self.assertEqual(len(original_edges), len(self.env.observation_space.graph.g_.edges(1)))
        self.assertEqual(0, len(self.env.observation_space.graph.g_.nodes[0]["_edges"]))
        self.assertEqual(0, len(self.env.observation_space.graph.g_.nodes[1]["_edges"]))

    def test_treatment_removes_infection_when_forced(self):
        # Arrange
        action_space = self._sut(treatment_conclusion_chance=1,
                                 treatment_recovery_rate_modifier=10)
        self.env.observation_space.graph.g_.nodes[1]["infected"] = 3
        self.env.observation_space.graph.g_.nodes[1]["status"].infected = True

        # Act
        cost = action_space.treat(env=self.env, target_node_id=1, step=1)

        # Assert
        self.assertEqual(action_space.treat_cost, cost)
        self.assertFalse(self.env.observation_space.graph.g_.nodes[0]["infected"] > 0)
        self.assertFalse(self.env.observation_space.graph.g_.nodes[1]["infected"] > 0)
        self.assertIsNone(self.env.observation_space.graph.g_.nodes[0]["status"].infected)
        self.assertFalse(self.env.observation_space.graph.g_.nodes[1]["status"].infected)
