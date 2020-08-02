from social_distancing_sim.environment.gym.gym_env import GymEnv
from social_distancing_sim.templates.undistanced import Undistanced


class SDSUndistanced(GymEnv):
    template = Undistanced
