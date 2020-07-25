import warnings

import gym


def register_test_envs():
    try:
        gym.envs.register(id='SDSTests-GymEnvFixedSeedFixture-v0',
                          entry_point='tests.common.env_fixtures.gym_env_fixed_seed_fixture:GymEnvFixedSeedFixture',
                          max_episode_steps=1000)
    except gym.error.Error as e:
        warnings.warn(str(e))

    try:
        gym.envs.register(id='SDSTests-GymEnvRandomSeedFixture-v0',
                          entry_point='tests.common.env_fixtures.gym_env_random_seed_fixture:GymEnvRandomSeedFixture',
                          max_episode_steps=1000)
    except gym.error.Error as e:
        warnings.warn(str(e))


def register_sim_test_envs():
    try:
        gym.envs.register(id='SDSTests-GymEnvDefaultFixture-v0',
                          entry_point='tests.common.env_fixtures.gym_envs_for_sim_tests:GymEnvDefaultFixture',
                          max_episode_steps=1000)
    except gym.error.Error as e:
        warnings.warn(str(e))

    try:
        gym.envs.register(id='SDSTests-GymEnvSpecifiedFixture-v0',
                          entry_point='tests.common.env_fixtures.gym_envs_for_sim_tests:GymEnvSpecifiedFixture',
                          max_episode_steps=1000)
    except gym.error.Error as e:
        warnings.warn(str(e))

    try:
        gym.envs.register(id='SDSTests-GymEnvSomePlottingFixture-v0',
                          entry_point='tests.common.env_fixtures.gym_envs_for_sim_tests:GymEnvSomePlottingFixture',
                          max_episode_steps=1000)
    except gym.error.Error as e:
        warnings.warn(str(e))

    try:
        gym.envs.register(id='SDSTests-GymEnvExtraPlottingFixture-v0',
                          entry_point='tests.common.env_fixtures.gym_envs_for_sim_tests:GymEnvExtraPlottingFixture',
                          max_episode_steps=1000)
    except gym.error.Error as e:
        warnings.warn(str(e))
