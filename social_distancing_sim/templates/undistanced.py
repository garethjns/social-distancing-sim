import social_distancing_sim.environment as env
from social_distancing_sim.templates.template_base import TemplateBase


class Undistanced(TemplateBase):
    @classmethod
    def build(cls, environment_seed: int = None,
              graph_seed: int = None,
              disease_seed: int = None,
              observation_space_seed: int = None) -> env.Environment:
        return env.Environment(name='exps/A herd of cats',
                               disease=env.Disease(name='COVID-19'),
                               observation_space=env.ObservationSpace(graph=env.Graph(community_n=40,
                                                                                      community_size_mean=16,
                                                                                      seed=graph_seed),
                                                                      test_rate=1),
                               environment_plotting=env.EnvironmentPlotting(ts_fields_g2=["Turn score"]))


if __name__ == "__main__":
    pop = Undistanced.build()
    pop.run(steps=100,
            plot=True)
