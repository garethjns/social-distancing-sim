import unittest
from typing import Any, Tuple

import gym

from social_distancing_sim.environment.gym.gym_env import GymEnv
from tests.common.env_fixtures import register_test_envs
from tests.common.env_fixtures.env_template_random_seed_fixture import EnvTemplateRandomSeedFixture


class TestGymEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_test_envs()

    @staticmethod
    def _run_for(env: GymEnv, steps: int = 5) -> Tuple[GymEnv, Any]:
        _ = env.reset()
        obs = None
        for _ in range(steps):
            obs, _, _, _ = env.step(([], []))

        return env, obs

    def test_import_registered_env(self):
        # Act
        env = gym.make('SDSTests-GymEnvFixedSeedFixture-v0')

        # Assert
        self.assertIsInstance(env.unwrapped, GymEnv)
        self.assertIsInstance(env, gym.wrappers.time_limit.TimeLimit)

    def test_create_with_on_the_fly_registration(self):
        # Act
        gym.envs.register(id='SDSTests-CustomEnv-v0',
                          entry_point='tests.unit.environment.gym.test_gym_env:_CustomEnv',
                          max_episode_steps=1000)
        env = gym.make('SDSTests-CustomEnv-v0')

        # Assert
        self.assertIsInstance(env.unwrapped, GymEnv)
        self.assertIsInstance(env, gym.wrappers.time_limit.TimeLimit)

    def test_works_with_standard_loop(self):
        # Arrange
        env = gym.make('SDSTests-GymEnvFixedSeedFixture-v0')

        # Act
        _ = self._run_for(env)

    def test_state_tuple_is_expected_shape(self):
        # Arrange
        env = gym.make('SDSTests-GymEnvFixedSeedFixture-v0')

        # Act
        obs = env.reset()

        # Assert
        self.assertEqual(3, len(obs))

    def test_agg_state_matches_internal_env(self):
        # Arrange
        env = gym.make('SDSTests-GymEnvFixedSeedFixture-v0')

        # Act
        env, obs = self._run_for(env)

        # Assert
        self.assertEqual((7,), obs[0].shape)
        self.assertEqual(obs[0][0], len(env.sds_env.observation_space.current_alive_nodes))
        self.assertEqual(obs[0][1], len(env.sds_env.observation_space.current_clear_nodes))
        self.assertEqual(obs[0][2], len(env.sds_env.observation_space.current_infected_nodes))
        self.assertEqual(obs[0][3], len(env.sds_env.observation_space.current_immune_nodes))
        self.assertEqual(obs[0][4], len(env.sds_env.observation_space.current_isolated_nodes))
        self.assertEqual(obs[0][5], len(env.sds_env.observation_space.current_masked_nodes))
        self.assertEqual(obs[0][6], len(env.sds_env.observation_space.unknown_nodes))

    def test_graph_matrix_state_matches_internal_env(self):
        # Arrange
        env = gym.make('SDSTests-GymEnvFixedSeedFixture-v0')

        # Act
        env, obs = self._run_for(env)

        # Assert
        self.assertEqual((env.sds_env.total_population, env.sds_env.total_population), obs[1].shape)

    def test_full_state_matches_internal_env(self):
        # Arrange
        env = gym.make('SDSTests-GymEnvFixedSeedFixture-v0')

        # Act
        env, obs = self._run_for(env)

        # Assert
        self.assertEqual((env.sds_env.total_population, 6), obs[2].shape)
        self.assertEqual(len(env.sds_env.observation_space.current_alive_nodes), obs[2][:, 0].sum())
        self.assertEqual(len(env.sds_env.observation_space.current_clear_nodes), obs[2][:, 1].sum())
        self.assertEqual(len(env.sds_env.observation_space.current_immune_nodes), obs[2][:, 3].sum())
        self.assertEqual(len(env.sds_env.observation_space.current_isolated_nodes), obs[2][:, 4].sum())
        self.assertEqual(len(env.sds_env.observation_space.current_masked_nodes), obs[2][:, 5].sum())
        # env.sds_env.observation_space.current_infected_nodes is excluded here as state can be different - it depends
        # on both test rate, and timeout of known infections. So Node status can be different from the currently
        # observed and the truth. (Status.state doesn't time out, it's only updated on retest).


class _CustomEnv(GymEnv):
    template = EnvTemplateRandomSeedFixture
