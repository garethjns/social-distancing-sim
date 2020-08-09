import unittest

from tests.common.env_fixtures.env_template_fixed_seed_fixture import EnvTemplateFixedSeedFixture


class TestEnvironment(unittest.TestCase):
    _template = EnvTemplateFixedSeedFixture()

    def setUp(self):
        self._env = self._template.build()
        for _ in range(30):
            self._env.step([], [])

    def test_empty_actions_targets(self):
        # Act
        observation, obs_turn_score, done = self._env.step([], [])

        # Assert
        self.assertIsInstance(obs_turn_score, float)
        self.assertIsInstance(done, bool)
        self.assertEqual(observation['completed_actions'], {})

    def test_actions_empty_targets(self):
        # Act
        observation, obs_turn_score, done = self._env.step([0, 1, 2, 3, 4, 5], [])

        # Assert
        self.assertIsInstance(obs_turn_score, float)
        self.assertIsInstance(done, bool)
        self.assertNotEqual(observation['completed_actions'], {})

    def test_actions_none_targets(self):
        # Act
        observation, obs_turn_score, done = self._env.step([0, 1, 2, 3, 4, 5], None)

        # Assert
        self.assertIsInstance(obs_turn_score, float)
        self.assertIsInstance(done, bool)
        self.assertNotEqual(observation['completed_actions'], {})

    def test_actions_valid_targets(self):
        # Arrange
        valid_treat_targets = self._env.observation_space.graph.current_infected_nodes[0:3]

        # Act
        observation, obs_turn_score, done = self._env.step([4, 4, 4], valid_treat_targets)

        # Assert
        self.assertIsInstance(obs_turn_score, float)
        self.assertIsInstance(done, bool)
        self.assertListEqual(list(observation['completed_actions'].values()), [4, 4, 4])

    def test_actions_invalid_targets(self):
        # Arrange
        invalid_treat_targets = self._env.observation_space.graph.current_clear_nodes[0:3]

        # Act
        observation, obs_turn_score, done = self._env.step([4, 4, 4], invalid_treat_targets)

        # Assert
        self.assertIsInstance(obs_turn_score, float)
        self.assertIsInstance(done, bool)
        # In this case, actions will still be completed, environment doesn't enforce sensible choices.
        self.assertListEqual(list(observation['completed_actions'].values()), [4, 4, 4])

    def test_clone_returns_same_env(self):
        # Arrange
        env1 = self._env

        # Act
        env2 = env1.clone()

        # Assert
        # Should match to initial conditions
        self.assertEqual(env1, env2)
        # But not on history or changes by stepping
        self.assertNotEqual(env1.history, env2.history)
