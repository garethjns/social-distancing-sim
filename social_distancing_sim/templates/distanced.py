import social_distancing_sim.environment as env
from social_distancing_sim.templates.template_base import TemplateBase


class Distanced(TemplateBase):
    @classmethod
    def build(cls, environment_seed: int = None,
              graph_seed: int = None,
              disease_seed: int = None,
              observation_space_seed: int = None) -> env.Environment:
        return env.Environment(name='exps/A socially responsible environments',
                               disease=env.Disease(name='COVID-19'),
                               observation_space=env.ObservationSpace(graph=env.Graph(community_n=40,
                                                                                      community_size_mean=16,
                                                                                      community_p_in=0.05,
                                                                                      community_p_out=0.04,
                                                                                      seed=graph_seed),
                                                                      test_rate=1),
                               environment_plotting=env.EnvironmentPlotting(ts_fields_g2=["Turn score"]))


if __name__ == "__main__":
    pop = Distanced.build()
    pop.run(steps=100,
            plot=True)
