from social_distancing_sim.gym.gym_env import GymEnv
from social_distancing_sim.templates.pop_746 import Pop746

from functools import partial

pop746 = partial(GymEnv, template=Pop746())


if __name__ == "__main__":
    a = pop746
    print(a)
    a.step({1: 'vaccinate'})
