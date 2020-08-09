"""
Run all the basic agents with a number of actions per turn using a MultiSim (n reps = 100). Doesn't save .gifs of
each rep, rather plots distributions of final scores.

Parameters here are similar to the visual version run in scripts/visual_compare_basic_agents.py.
"""
from typing import List

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

import social_distancing_sim.agent as agent
import social_distancing_sim.environment as env
import social_distancing_sim.sim as sim
from social_distancing_sim.environment.gym.gym_env import GymEnv
from social_distancing_sim.templates.template_base import TemplateBase


def plot_dists(multi_sims: List[sim.MultiSim],
               result: str = "Overall score") -> plt.Figure:
    """Plot final score distributions across repetitions, for all agents."""
    fig, axs = plt.subplots(nrows=6,
                            ncols=1,
                            figsize=(8, 10))
    ax_map = {'DummyAgent': axs[0],
              'RandomAgent': axs[1],
              'VaccinationAgent': axs[2],
              'IsolationAgent': axs[3],
              'TreatmentAgent': axs[4],
              'MaskingAgent': axs[5]}

    min_score = np.inf
    max_score = -np.inf
    for run in multi_sims:
        min_score = min(min_score, run.results[result].min())
        max_score = max(max_score, run.results[result].max())

        sns.distplot(run.results[result],
                     hist=False,
                     ax=ax_map[type(run.sim.agent).__name__],
                     label=run.sim.agent.actions_per_turn)

    for ax_i, (ax_name, ax) in enumerate(ax_map.items()):
        ax.set_title(ax_name, fontweight='bold')
        ax.set_xlim([min_score - abs(min_score * 0.2), max_score + abs(max_score * 0.2)])
        if ax_i != 3:
            ax.set_xlabel('')
        else:
            ax.set_xlabel(ax.get_xlabel(),
                          fontweight='bold')
        ax.set_ylabel('Prop.',
                      fontweight='bold')
        ax.legend(title='n actions / turn')

    return fig


class EnvTemplate(TemplateBase):
    def build(self):
        env_ = env.Environment(name=f"stats_compare_basic_agents_custom_env",
                               action_space=env.ActionSpace(nothing_cost=0,
                                                            vaccinate_cost=0,
                                                            isolate_cost=0,
                                                            reconnect_cost=0,
                                                            treat_cost=0,
                                                            mask_cost=0),
                               disease=env.Disease(name='COVID-19',
                                                   virulence=0.01,
                                                   immunity_mean=0.95,
                                                   recovery_rate=0.9,
                                                   immunity_decay_mean=0.005),
                               healthcare=env.Healthcare(capacity=50),
                               observation_space=env.ObservationSpace(graph=env.Graph(community_n=15,
                                                                                      community_size_mean=10,
                                                                                      seed=None),
                                                                      test_rate=1,
                                                                      seed=None),
                               initial_infections=15,
                               seed=None)
        return env_


class CustomEnv(GymEnv):
    template = EnvTemplate()


if __name__ == "__main__":
    env_name = f"SDSTests-CustomEnv{np.random.randint(2e6)}-v0"
    gym.envs.register(id=env_name,
                      entry_point='scripts.stats_compare_basic_agents:CustomEnv',
                      max_episode_steps=1000)
    env_spec = gym.make(env_name).spec

    # For a specified parameter set....
    agents = [agent.DummyAgent, agent.RandomAgent, agent.VaccinationAgent, agent.IsolationAgent,
              agent.TreatmentAgent, agent.MaskingAgent]
    n_actions = [3, 6, 12, 24]
    multi_sims = []

    # Create all Agent, Environments, Sims, and MultiSims. The MultiSim will run the Sim multiple times without
    # visualisations
    for n_act, agt in np.array(np.meshgrid(n_actions, agents)).T.reshape(-1, 2):
        agt_ = agt(actions_per_turn=n_act, name=f"{agt.__name__} - {n_act} actions")
        sim_ = sim.Sim(env_spec=env_spec, agent=agt_, n_steps=125)

        multi_sims.append(sim.MultiSim(sim_, name='basic agent comparison',
                                       n_reps=300, n_jobs=60))

    # Run all the sims. No need to parallelize here as it's done across n reps in MultiSim.run()
    for ms in tqdm(multi_sims):
        ms.run()

    fig = plot_dists(multi_sims, "Overall score")
    plt.show()
    fig.savefig('basic_agent_comparison_score.png')

    fig = plot_dists(multi_sims, "Total deaths")
    plt.show()
    fig.savefig('basic_agent_comparison_deaths.png')
