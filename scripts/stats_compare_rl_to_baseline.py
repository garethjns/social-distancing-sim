"""Run multiple simulations using some basic agents, some policy agents, and a trained linear q learner."""

from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import social_distancing_sim.agent as agent
import social_distancing_sim.sim as sim
from social_distancing_sim.gym.agent.rl.q_learners.linear_q_agent import LinearQAgent
from social_distancing_sim.templates.small import Small


def plot_dists(multi_sims: List[sim.MultiSim],
               result: str = "Overall score") -> plt.Figure:
    """Plot final score distributions across repetitions, for all agents."""
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

    ax.set_title("Agent comparison",
                 fontweight='bold')
    ax.set_xlim([min_score - abs(min_score * 0.2), max_score + abs(max_score * 0.2)])
    ax.set_xlabel(ax.get_xlabel(),
                  fontweight='bold')
    ax.set_ylabel('Prop.',
                  fontweight='bold')
    ax.legend(title='Agent')

    return fig


if __name__ == "__main__":
    steps = 200
    distancing_params = {"actions_per_turn": 15,
                         "start_step": {'isolate': 15, 'reconnect': 60},
                         "end_step": {'isolate': 55, 'reconnect': steps}}
    vaccination_params = {"actions_per_turn": 5,
                          "start_step": {'vaccinate': 60},
                          "end_step": {'vaccinate': steps}}
    treatment_params = {"actions_per_turn": 5,
                        "start_step": {'treat': 50},
                        "end_step": {'treat': steps}}

    linear_q_agent = LinearQAgent.load('linear_q_learner.pkl')
    linear_q_agent.name = 'linear_q_agent'

    agents = [agent.RandomAgent(),
              agent.MultiAgent(name="Distancing, vaccination, treatment",
                               agents=[agent.DistancingPolicyAgent(**distancing_params),
                                       agent.VaccinationPolicyAgent(**vaccination_params),
                                       agent.TreatmentPolicyAgent(**treatment_params)]),
              linear_q_agent, ]

    # Loop over the parameter set and create the Agents, Environments, and the Sim handler
    multi_sims = []
    for agt in agents:
        # Name the environment according to the agent used
        sim_ = sim.Sim(env=Small().build(),
                       tqdm_on=False,
                       agent=agt,
                       n_steps=steps)

        multi_sims.append(sim.MultiSim(sim_,
                                       name='rl agents',
                                       n_reps=100))

    # Run all the sims. No need to parallelize here as it's done across n reps in MultiSim.run()
    for ms in tqdm(multi_sims):
        ms.run()

    fig = plot_dists(multi_sims, "Overall score")
    plt.show()
    fig.savefig('linear_q_agent_comparison_score.png')

    fig = plot_dists(multi_sims, "Total deaths")
    plt.show()
    fig.savefig('linear_q_agent_comparison_deaths.png')
