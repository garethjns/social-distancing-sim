from social_distancing_sim.gym.gym_env import GymEnv
from social_distancing_sim.templates.pop_746 import Pop746
from social_distancing_sim.templates.small import Small

from functools import partial

pop746 = partial(GymEnv, template=Pop746())
small = partial(GymEnv, template=Small())
