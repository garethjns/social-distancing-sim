import unittest
from unittest.mock import MagicMock

from social_distancing_sim.environment import Status
from social_distancing_sim.environment.action_space import ActionSpace
from social_distancing_sim.environment.graph import Graph


class TestActionSpace(unittest.TestCase):
    _sut = ActionSpace()
    _implemented_actions = ['nothing', 'vaccinate', 'isolate', 'reconnect', 'treat', 'provide_mask']
    _implemented_action_ids = [0, 1, 2, 3, 4, 5]

    @staticmethod
    def _build_mock_env() -> MagicMock:
        env = MagicMock()
        env.disease = MagicMock()
        env.observation_space = MagicMock()
        env.observation_space.graph = MagicMock()
        env.observation_space.graph.g_ = MagicMock()
        env.observation_space.graph.g_.nodes = {0: {'status': Status()}}

        return env

    def test_expected_default_actions_are_available(self):
        # Act
        available_actions = self._sut.available_actions

        # Assert
        self.assertListEqual(self._implemented_action_ids, available_actions)

    def test_n_returns_expected_n_actions(self):
        # Act
        n_available_actions = self._sut.n

        # Assert
        self.assertEqual(len(self._implemented_actions), n_available_actions)

    def test_sampled_returns_valid_action(self):
        # Act
        action = self._sut.sample()

        # Assert
        self.assertIn(action, self._implemented_action_ids)

    def test_name_to_id(self):
        # Act
        action_id = self._sut.get_action_id('isolate')

        # Assert
        self.assertEqual(self._implemented_actions.index('isolate'), action_id)

    def test_id_to_name(self):
        # Act
        action_name = self._sut.get_action_name(3)

        # Assert
        self.assertEqual(action_name, self._implemented_actions[3])

    def test_select_n_random_actions_with_larger_pool(self):
        # Act
        targets = self._sut.select_random_target(n=2,
                                                 available_targets=[1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Assert
        self.assertEqual(2, len(targets))
        self.assertNotIn(-1, targets)

    def test_select_n_random_actions_with_lesser_pool(self):
        # Act
        targets = self._sut.select_random_target(n=12,
                                                 available_targets=[1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Assert
        self.assertEqual(12, len(targets))
        self.assertEqual(9, len([t for t in targets if t != -1]))
        self.assertEqual(3, len([t for t in targets if t == -1]))

    def test_vaccinate_node_adds_immunity_via_disease(self):
        # Arrange
        env = self._build_mock_env()

        def _mock_give_immunity(node, immunity=0.6):
            node["immune"] = immunity
            return node

        # Mock Disease.give_immunity, which takes and node and returns a node
        env.disease.give_immunity = _mock_give_immunity

        # Act
        step = 0
        cost = self._sut.vaccinate(env=env, target_node_id=0, step=step)

        # Assert
        self.assertEqual(step, env.observation_space.graph.g_.nodes[0]['last_tested'])
        self.assertGreater(env.observation_space.graph.g_.nodes[0]['immune'], 0)
        self.assertEqual(self._sut.vaccinate_cost, cost)

    def test_provide_mask_to_node_adds_mask_via_graph(self):
        # Arrange
        # This is harder to mock as graph methods take the whole graph as input, not just the node (like disease)
        # Just using a default graph for now.
        env = self._build_mock_env()
        env.observation_space.graph = Graph()
        env.observation_space.graph.g_.nodes[0]["status"] = MagicMock()

        # Act
        step = 0
        cost = self._sut.provide_mask(env=env, target_node_id=0, step=step)

        self.assertEqual(self._sut.mask_cost, cost)
        self.assertGreater(env.observation_space.graph.g_.nodes[0]['mask'], 0)
