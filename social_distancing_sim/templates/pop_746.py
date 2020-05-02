import social_distancing_sim.environment as env
from social_distancing_sim.templates.template_base import TemplateBase


class Pop746(TemplateBase):
    """Base environment template specifying defaults."""

    @classmethod
    def build(cls, environment_seed: int = None,
              graph_seed: int = 20200423,
              disease_seed: int = None,
              observation_space_seed: int = None) -> env.Environment:
        return env.Environment(name="example environment",
                               disease=env.Disease(name='COVID-19',
                                                   virulence=0.006,
                                                   immunity_mean=0.6,
                                                   immunity_decay_mean=0.15,
                                                   seed=disease_seed),
                               healthcare=env.Healthcare(),
                               observation_space=env.ObservationSpace(graph=env.Graph(community_n=50,
                                                                                      community_size_mean=15,
                                                                                      community_p_in=0.1,
                                                                                      community_p_out=0.05,
                                                                                      seed=graph_seed),
                                                                      test_rate=0.05,
                                                                      seed=observation_space_seed),
                               seed=environment_seed)
