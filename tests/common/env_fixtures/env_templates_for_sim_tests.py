import social_distancing_sim.environment as env
from social_distancing_sim.environment import Disease, Healthcare, ObservationSpace, Graph, EnvironmentPlotting

from social_distancing_sim.templates.template_base import TemplateBase

COMMON_SETUP = {'disease': Disease(name='COVID-19',
                                   virulence=0.01,
                                   seed=123,
                                   immunity_mean=0.95,
                                   immunity_decay_mean=0.05),
                'healthcare': Healthcare(capacity=5),
                'observation_space': ObservationSpace(graph=Graph(community_n=15,
                                                                  community_size_mean=10,
                                                                  seed=123 + 1),
                                                      test_rate=1,
                                                      seed=123 + 2),
                'seed': 123 + 3}

PLOT_SET_1 = ["Score", "Action cost", "Overall score"]
PLOT_SET_2 = ["Observed Score", "Action cost", "Observed overall score"]


class DefaultEnvTemplate(TemplateBase):
    @classmethod
    def build(cls) -> env.Environment:
        return env.Environment(name='default_env', disease=env.Disease(),
                               healthcare=env.Healthcare(),
                               observation_space=env.ObservationSpace(graph=env.Graph()))


class SpecifiedEnvTemplate(TemplateBase):
    @classmethod
    def build(cls) -> env.Environment:
        return env.Environment(name='specific_env', **COMMON_SETUP)


class SomePlottingEnvTemplate(TemplateBase):
    @classmethod
    def build(cls) -> env.Environment:
        return env.Environment(name='some_plotting_env', **COMMON_SETUP,
                               environment_plotting=EnvironmentPlotting(ts_fields_g2=PLOT_SET_1))


class ExtraPlottingEnvTemplate(TemplateBase):
    @classmethod
    def build(cls) -> env.Environment:
        return env.Environment(name='extra_plotting_env', **COMMON_SETUP,
                               environment_plotting=EnvironmentPlotting(ts_fields_g2=PLOT_SET_1,
                                                                        ts_obs_fields_g2=PLOT_SET_2))
