from social_distancing_sim.environment.gym.gym_env import GymEnv
from tests.common.env_fixtures.env_template_fixed_seed_fixture import EnvTemplateFixedSeedFixture


class GymEnvFixedSeedFixture(GymEnv):
    """Registered on import from .env_fixtures."""
    template = EnvTemplateFixedSeedFixture
