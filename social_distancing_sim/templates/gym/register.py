import warnings

import gym


def register(name: str, entry_point: str, max_episode_steps: int = 1000):
    try:
        gym.envs.register(id=name, entry_point=entry_point, max_episode_steps=max_episode_steps)
    except gym.error.Error as e:
        warnings.warn(str(e))


def register_template_envs():
    register('SDS-746-v0', 'social_distancing_sim.templates.gym.sds_746:SDS746')
    register('SDS-small-v0', 'social_distancing_sim.templates.gym.sds_small:SDSSmall')
    register('SDS-distanced-v0', 'social_distancing_sim.templates.gym.sds_distanced:SDSDistanced')
    register('SDS-undistanced-v0', 'social_distancing_sim.templates.gym.sds_undistanced:SDSUndistanced')
