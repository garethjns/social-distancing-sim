import social_distancing_sim.environment as env
from social_distancing_sim.environment.gym.gym_env import GymEnv
from social_distancing_sim.templates.template_base import TemplateBase


class EnvTemplate(TemplateBase):
    def build(self):
        return env.Environment(name="example_sim_env",
                               action_space=env.ActionSpace(),
                               disease=env.Disease(),
                               healthcare=env.Healthcare(),
                               environment_plotting=env.EnvironmentPlotting(ts_fields_g2=["Actions taken",
                                                                                          "Overall score"]),
                               observation_space=env.ObservationSpace(
                                   graph=env.Graph(community_n=20,
                                                   community_size_mean=15,
                                                   considered_immune_threshold=0.7),
                                   test_rate=1),
                               initial_infections=15)


# Define an gym compatible environment environment
class CustomEnv(GymEnv):
    template = EnvTemplate()
