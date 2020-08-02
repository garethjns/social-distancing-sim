from social_distancing_sim.environment.gym.gym_env import GymEnv
from social_distancing_sim.templates.distanced import Distanced


class SDSDistanced(GymEnv):
    template = Distanced
