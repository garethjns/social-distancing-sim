"""Run all the basic policy agents with a number of actions per turn (n reps = 1). Generate and save .gif."""

from functools import partial

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
        return env.Environment(name="visual_compare_policy_agents_custom_env",
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

    # Prepare a custom environment
    env_name = f"SDSTests-CustomEnv{np.random.randint(2e6)}-v0"
    gym.envs.register(id=env_name,
                      entry_point='scripts.visual_compare_policy_agents:CustomEnv',
                      max_episode_steps=1000)
    env_spec = gym.make(env_name).spec

    # Create a parameter set containing all combinations of the 3 policy agents, and a small set of n_actions
    agents = [agent.DummyAgent,
              partial(agent.DistancingPolicyAgent,
                      start_step={'isolate': 10,
                                  'reconnect': 50},
                      end_step={'isolate': 40,
                                'reconnect': 60}),
              partial(agent.VaccinationPolicyAgent,
                      start_step={'vaccinate': 20},
                      end_step={'vaccinate': 40}),
              partial(agent.TreatmentPolicyAgent,
                      start_step={'treat': 20},
                      end_step={'treat': 40}),
              ]
    n_actions = [10, 20]
    sims = []

    # Loop over the parameter set and create the Agents, Environments, and the Sim handler
    for n_act, agt in np.array(np.meshgrid(n_actions,
                                           agents)).T.reshape(-1, 2):
        agt_ = agt(actions_per_turn=n_act,
                   name=f"{agt.func.__name__ if isinstance(agt, partial) else agt.__name__} - {n_act} actions")

        sims.append(sim.Sim(env_spec=env_spec, agent=agt_, n_steps=300,
                            plot=False, save=True, tqdm_on=True))  # Show progress bars for running sims

    # Run all the prepared Sims
    Parallel(n_jobs=60, backend='loky')(delayed(run_and_replay)(sim) for sim in sims)
