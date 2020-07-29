from social_distancing_sim.environment.gym.gym_env import GymEnv
from tests.common.env_fixtures.env_templates_for_sim_tests import DefaultEnvTemplate, SpecifiedEnvTemplate, \
    ExtraPlottingEnvTemplate, SomePlottingEnvTemplate


class GymEnvDefaultFixture(GymEnv):
    template = DefaultEnvTemplate


class GymEnvSpecifiedFixture(GymEnv):
    template = SpecifiedEnvTemplate


class GymEnvSomePlottingFixture(GymEnv):
    template = SomePlottingEnvTemplate


class GymEnvExtraPlottingFixture(GymEnv):
    template = ExtraPlottingEnvTemplate
