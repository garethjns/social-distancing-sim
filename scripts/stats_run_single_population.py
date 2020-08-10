import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import social_distancing_sim.environment as env
from social_distancing_sim.environment.gym.gym_env import GymEnv
from social_distancing_sim.sim import MultiSim, Sim
from social_distancing_sim.templates.template_base import TemplateBase


class EnvTemplate(TemplateBase):
    def build(self):
        return env.Environment(name="visual_run_simulation_with_agent_custom_env",
                               action_space=env.ActionSpace(isolate_efficiency=0.5,
                                                            vaccinate_efficiency=0.95),
                               disease=env.Disease(name='COVID-19',
                                                   virulence=0.008,
                                                   immunity_mean=0.7,
                                                   recovery_rate=0.9,
                                                   immunity_decay_mean=0.01),
                               healthcare=env.Healthcare(capacity=75),
                               environment_plotting=env.EnvironmentPlotting(ts_fields_g2=["Actions taken",
                                                                                          "Overall score"]),
                               observation_space=env.ObservationSpace(
                                   graph=env.Graph(community_n=20,
                                                   community_size_mean=15,
                                                   considered_immune_threshold=0.7),
                                   test_rate=1),
                               initial_infections=15)


class CustomEnv(GymEnv):
    template = EnvTemplate()


if __name__ == "__main__":
    # Prepare a custom environment
    env_name = f"SDSTests-CustomEnv{np.random.randint(2e6)}-v0"
    gym.envs.register(id=env_name,
                      entry_point='scripts.stats_run_single_population:CustomEnv',
                      max_episode_steps=1000)

    sim = Sim(env_spec=gym.make(env_name).spec)

    ms = MultiSim(sim, n_reps=10, n_jobs=50)

    ms.run()

    sns.distplot(ms.results['Overall score'])
    sns.distplot(ms.results['Total deaths'])
    plt.show()
