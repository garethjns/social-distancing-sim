import unittest

import gym

from social_distancing_sim.environment.gym.gym_env import GymEnv
from tests.common.env_fixtures import register_test_envs
from tests.common.env_fixtures.env_template_random_seed_fixture import EnvTemplateRandomSeedFixture


class TestGymEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_test_envs()

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
        _ = env.reset()
        for _ in range(5):
            env.step([])


class _CustomEnv(GymEnv):
    template = EnvTemplateRandomSeedFixture
