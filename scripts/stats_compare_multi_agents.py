"""
Run all the basic agents with a number of actions per turn using a MultiSim (n reps = 100). Doesn't save .gifs of
each rep, rather plots distributions of final scores.

Parameters here are similar to the visual version run in scripts/visual_compare_multi_agents.py.
"""
from typing import List

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

import social_distancing_sim.environment as env
import social_distancing_sim.sim as sim
from scripts.visual_compare_multi_agents import AGENTS
from social_distancing_sim.environment.gym.gym_env import GymEnv
from social_distancing_sim.templates.template_base import TemplateBase

SEED = 123
STEPS = 250


def plot_dists(multi_sims: List[sim.MultiSim],
               result: str = "Overall score") -> plt.Figure:
    """Plot final score distributions across repetitions, for all agents."""
    sns.set_palette(sns.cubehelix_palette(len(multi_sims)))
    fig, ax = plt.subplots(nrows=1,
                           ncols=1,
                           figsize=(8, 8))

    min_score = 0
    max_score = 0
    for run in multi_sims:
        min_score = min(min_score, run.results[result].min())
        max_score = max(max_score, run.results[result].max())

        sns.distplot(run.results[result],
                     hist=False,
                     label=run.sim.agent.name)

    ax.set_title("Policy comparison",
                 fontweight='bold')
    ax.set_xlim([min_score - abs(min_score * 0.2), max_score + abs(max_score * 0.2)])
    ax.set_xlabel(ax.get_xlabel(),
                  fontweight='bold')
    ax.set_ylabel('Prop.',
                  fontweight='bold')
    ax.legend(title='Agent')

    return fig


class EnvTemplate(TemplateBase):
    def build(self):
        env_ = env.Environment(name=f"stats_compare_multi_agents_custom_env",
                               action_space=env.ActionSpace(vaccinate_cost=0,
                                                            treat_cost=0,
                                                            isolate_cost=0,
                                                            isolate_efficiency=0.70,
                                                            reconnect_efficiency=0.2,
                                                            treatment_conclusion_chance=0.5,
                                                            treatment_recovery_rate_modifier=1.8,
                                                            vaccinate_efficiency=1.25),
                               disease=env.Disease(name='COVID-19',
                                                   virulence=0.0055,
                                                   seed=None,
                                                   immunity_mean=0.7,
                                                   recovery_rate=0.9,
                                                   immunity_decay_mean=0.01),
                               healthcare=env.Healthcare(capacity=200),
                               observation_space=env.ObservationSpace(
                                   graph=env.Graph(community_n=30,
                                                   community_size_mean=20,
                                                   community_p_out=0.08,
                                                   community_p_in=0.16,
                                                   seed=None),
                                   test_rate=1,
                                   seed=None),
                               initial_infections=5,
                               random_infection_chance=1,
                               seed=None)
        return env_


class CustomEnv(GymEnv):
    template = EnvTemplate()


def run_multi_sims():
    env_name = f"SDSTests-CustomEnv{np.random.randint(2e6)}-v0"
    gym.envs.register(id=env_name,
                      entry_point='scripts.stats_compare_multi_agents:CustomEnv',
                      max_episode_steps=1000)
    env_spec = gym.make(env_name).spec

    # Loop over the parameter set and create the Agents, Environments, and the Sim handler
    multi_sims = []
    for agt in AGENTS:
        # Name the environments according to the agent used
        sim_ = sim.Sim(env_spec=env_spec, agent=agt, n_steps=150)

        multi_sims.append(sim.MultiSim(sim_,
                                       n_jobs=30,
                                       name='policy agent comparison',
                                       n_reps=300))

    # Run all the sims. No need to parallelize here as it's done across n reps in MultiSim.run()
    for ms in tqdm(multi_sims):
        ms.run()

    return multi_sims


if __name__ == "__main__":
    multi_sims_ = run_multi_sims()

    fig = plot_dists(multi_sims_, "Overall score")
    plt.show()
    fig.savefig('multi_agent_comparison_score.png')

    fig = plot_dists(multi_sims_, "Total deaths")
    plt.show()
    fig.savefig('multi_agent_comparison_deaths.png')
