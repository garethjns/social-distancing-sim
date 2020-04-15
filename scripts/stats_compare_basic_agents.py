"""
Run all the basic agents with a number of actions per turn using a MultiSim (n reps = 100). Doesn't save .gifs of
each rep, rather plots distributions of final scores.

Parameters here match the visual version run in scripts/stats_compare_basic_agents.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

import social_distancing_sim.agent as agent
import social_distancing_sim.environment as env
import social_distancing_sim.sim as sim


def plot_dists(result: str = "Overall score") -> plt.Figure:
    """Plot final score distributions across repetitions, for all agents."""
    fig, axs = plt.subplots(nrows=4,
                            ncols=1,
                            figsize=(8, 8))
    ax_map = {'DummyAgent': axs[0],
              'RandomAgent': axs[1],
              'VaccinationAgent': axs[2],
              'IsolationAgent': axs[3]}

    min_score = 0
    max_score = 0
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


if __name__ == "__main__":

    # For a specified parameter set....
    agents = [agent.DummyAgent, agent.RandomAgent, agent.VaccinationAgent, agent.IsolationAgent]
    n_actions = [3, 6, 12, 24]
    multi_sims = []

    # Create all Agent, Environments, Sims, and MultiSims. The MultiSim will run the Sim multiple times without
    # visualisations
    for n_act, agt in np.array(np.meshgrid(n_actions,
                                           agents)).T.reshape(-1, 2):
        agt_ = agt(actions_per_turn=n_act)
        env_ = env.Environment(name=f"{type(agt_).__name__} - {n_act} actions",
                               disease=env.Disease(name='COVID-19',
                                                   virulence=0.01,
                                                   seed=None,
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
                               seed=None)  # Use a different seed on every run)

        sim_ = sim.Sim(env=env_,
                       agent=agt_,
                       n_steps=125)

        multi_sims.append(sim.MultiSim(sim_,
                                       name='basic agent comparison 2',
                                       n_reps=100))

    # Run all the sims. No need to parallelize here as it's done across n reps in MultiSim.run()
    for ms in tqdm(multi_sims):
        ms.run()

    fig = plot_dists("Overall score")
    plt.show()
    fig.savefig('agent_comparison_score.png')

    fig = plot_dists("Total deaths")
    plt.show()
    fig.savefig('agent_comparison_deaths.png')
