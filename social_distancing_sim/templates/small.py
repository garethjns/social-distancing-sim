import social_distancing_sim.environment as env
from social_distancing_sim.templates.template_base import TemplateBase


class Small(TemplateBase):
    """Base environment template specifying defaults."""

    @classmethod
    def build(cls, environment_seed: int = None,
              graph_seed: int = 20200429,
              disease_seed: int = None,
              observation_space_seed: int = None) -> env.Environment:
        return env.Environment(name="example environment",
                               initial_infections=4,
                               action_space=env.ActionSpace(vaccinate_cost=0.1,
                                                            isolate_cost=0.1,
                                                            treat_cost=0.08,
                                                            reconnect_cost=0,
                                                            mask_cost=0.05),
                               scoring=env.Scoring(clear_yield_per_edge=0.2,
                                                   infection_penalty=-0.25,
                                                   death_penalty=-10),
                               disease=env.Disease(name='COVID-19',
                                                   virulence=0.05,
                                                   immunity_mean=0.6,
                                                   immunity_decay_mean=0.15,
                                                   seed=disease_seed),
                               environment_plotting=env.EnvironmentPlotting(ts_fields_g2=['Overall score']),
                               healthcare=env.Healthcare(capacity=8),
                               observation_space=env.ObservationSpace(graph=env.Graph(community_n=8,
                                                                                      community_size_mean=4,
                                                                                      community_p_in=0.8,
                                                                                      community_p_out=0.2,
                                                                                      seed=graph_seed),
                                                                      test_rate=1,
                                                                      seed=observation_space_seed),
                               seed=environment_seed)


if __name__ == "__main__":
    pop = Small.build()
    pop.run(steps=100,
            plot=True)
