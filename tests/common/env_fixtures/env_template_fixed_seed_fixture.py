import social_distancing_sim.environment as env
from social_distancing_sim.templates.template_base import TemplateBase


class EnvTemplateFixedSeedFixture(TemplateBase):
    @classmethod
    def build(cls) -> env.Environment:
        return env.Environment(name="env_template_fixed_seed_fixture",
                               disease=env.Disease(name='COVID-19',
                                                   virulence=0.1,
                                                   immunity_mean=0.6,
                                                   immunity_decay_mean=0.15,
                                                   seed=111),
                               healthcare=env.Healthcare(),
                               observation_space=env.ObservationSpace(graph=env.Graph(community_n=5,
                                                                                      community_size_mean=5,
                                                                                      community_p_in=1,
                                                                                      community_p_out=0.5,
                                                                                      seed=222),
                                                                      test_rate=0.05,
                                                                      seed=333),
                               seed=444)
