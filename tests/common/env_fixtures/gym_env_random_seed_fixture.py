from social_distancing_sim.environment.gym.gym_env import GymEnv
from tests.common.env_fixtures.env_template_random_seed_fixture import EnvTemplateRandomSeedFixture


class GymEnvRandomSeedFixture(GymEnv):
    """Registered on import from .env_fixtures."""
    template = EnvTemplateRandomSeedFixture
