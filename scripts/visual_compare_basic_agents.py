"""
Run all the basic agents with a number of actions per turn (n reps = 1). Generate and save .gif.

Parameters here match the stats version run in scripts/stats_compare_basic_agents.py.
"""
import gym
import numpy as np
from joblib import Parallel, delayed

import social_distancing_sim.agent as agent
import social_distancing_sim.environment as env
import social_distancing_sim.sim as sim
from social_distancing_sim.environment.gym.gym_env import GymEnv
from social_distancing_sim.templates.template_base import TemplateBase


def run_and_replay(sim):
    sim.run()
    if sim.save:
        sim.env.replay()


class EnvTemplate(TemplateBase):
    def build(self):
        seed = 123
        return env.Environment(name=f"visual_compare_basic_agents_custom_env",
                               disease=env.Disease(name='COVID-19',
                                                   virulence=0.01,
                                                   seed=seed,
                                                   immunity_mean=0.95,
                                                   recovery_rate=0.9,
                                                   immunity_decay_mean=0.005),
                               healthcare=env.Healthcare(capacity=50),
                               environment_plotting=env.EnvironmentPlotting(ts_fields_g2=["Turn score", "Action cost",
                                                                                          "Overall score"]),
                               observation_space=env.ObservationSpace(graph=env.Graph(community_n=15,
                                                                                      community_size_mean=10,
                                                                                      seed=seed + 1),

                                                                      test_rate=1,
                                                                      seed=seed + 2),
                               initial_infections=15,
                               seed=seed + 3)


class CustomEnv(GymEnv):
    template = EnvTemplate()


if __name__ == "__main__":

    # Prepare a custom environment
    env_name = f"SDS-CustomEnv{np.random.randint(2e6)}-v0"
    gym.envs.register(id=env_name,
                      entry_point='scripts.visual_compare_basic_agents:CustomEnv',
                      max_episode_steps=1000)
    env_spec = gym.make(env_name).spec

    # Prepare agents
    agents = [agent.DummyAgent, agent.RandomAgent, agent.VaccinationAgent, agent.IsolationAgent, agent.TreatmentAgent,
              agent.MaskingAgent]
    n_actions = [3, 6, 12]

    # Prepare Sims
    sims = []
    for n_act, agt in np.array(np.meshgrid(n_actions, agents)).T.reshape(-1, 2):
        agt_ = agt(actions_per_turn=n_act, name=f"{agt.__name__} - {n_act} actions")
        sims.append(sim.Sim(env_spec=env_spec, agent=agt_,
                            n_steps=125, plot=False, save=True,
                            tqdm_on=True, logging=True))  # Show progress bars for running sims

    # Run all the prepared Sims
    Parallel(n_jobs=-1, backend='loky')(delayed(run_and_replay)(sim) for sim in sims)
