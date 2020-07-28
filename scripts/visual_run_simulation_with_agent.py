import gym
import numpy as np

import social_distancing_sim.environment as env
import social_distancing_sim.sim as sim
from social_distancing_sim.agent.basic_agents.vaccination_agent import VaccinationAgent
from social_distancing_sim.environment.gym.gym_env import GymEnv
from social_distancing_sim.templates.template_base import TemplateBase


class EnvTemplate(TemplateBase):
    def build(self):
        seed = 123
        return env.Environment(name="visual_run_simulation_with_agent_custom_env",
                               action_space=env.ActionSpace(isolate_efficiency=0.5,
                                                            vaccinate_efficiency=0.95),
                               disease=env.Disease(name='COVID-19',
                                                   virulence=0.008,
                                                   seed=seed,
                                                   immunity_mean=0.7,
                                                   recovery_rate=0.9,
                                                   immunity_decay_mean=0.01),
                               healthcare=env.Healthcare(capacity=75),
                               environment_plotting=env.EnvironmentPlotting(ts_fields_g2=["Actions taken",
                                                                                          "Overall score"]),
                               observation_space=env.ObservationSpace(
                                   graph=env.Graph(community_n=20,
                                                   community_size_mean=15,
                                                   considered_immune_threshold=0.7,
                                                   seed=seed + 1),
                                   test_rate=1,
                                   seed=seed + 2),
                               initial_infections=15,
                               seed=seed + 3)


class CustomEnv(GymEnv):
    template = EnvTemplate()


if __name__ == "__main__":
    seed = 123

    # Prepare a custom environment
    env_name = f"SDSTests-CustomEnv{np.random.randint(2e6)}-v0"
    gym.envs.register(id=env_name,
                      entry_point='scripts.visual_run_simulation_with_agent:CustomEnv',
                      max_episode_steps=1000)
    env_spec = gym.make(env_name).spec

    sim = sim.Sim(env_spec=env_spec,
                  agent=VaccinationAgent(actions_per_turn=15,
                                         seed=seed),
                  plot=True,
                  save=True,
                  tqdm_on=True)

    sim.run()
