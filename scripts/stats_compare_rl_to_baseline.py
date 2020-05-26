"""Run multiple simulations using some basic agents, some policy agents, and a trained linear q learner."""

from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import social_distancing_sim.agent as agent
import social_distancing_sim.sim as sim
from scripts.visual_compare_multi_agents import DISTANCING_PARAMS, MASKING_PARAMS, TREATMENT_PARAMS, VACCINATION_PARAMS
from social_distancing_sim.gym.agent.rl.q_learners.deep_q_agent import DeepQAgent
from social_distancing_sim.gym.agent.rl.q_learners.linear_q_agent import LinearQAgent
from social_distancing_sim.templates.small import Small


def prepare_tf(memory_limit: int = 1024):
    import tensorflow as tf

    tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=memory_limit)])


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
    prepare_tf()

    steps = 200

    linear_q_agent = LinearQAgent.load('linear_q_learner.pkl')
    linear_q_agent.name = 'linear_q_agent'
    deep_q_agent = DeepQAgent.load('deep_q_learner.pkl')
    deep_q_agent.name = 'deep_q_agent'

    agents = [agent.DummyAgent(name='Dummy agent'),
              agent.MultiAgent(name="Distancing, vaccination, treatment",
                               agents=[agent.DistancingPolicyAgent(**DISTANCING_PARAMS),
                                       agent.VaccinationPolicyAgent(**VACCINATION_PARAMS),
                                       agent.TreatmentPolicyAgent(**TREATMENT_PARAMS)]),
              agent.MultiAgent(name="Distancing, vaccination, treatment, masking",
                               agents=[agent.DistancingPolicyAgent(**DISTANCING_PARAMS),
                                       agent.VaccinationPolicyAgent(**VACCINATION_PARAMS),
                                       agent.TreatmentPolicyAgent(**TREATMENT_PARAMS),
                                       agent.MaskingPolicyAgent(**MASKING_PARAMS)]),
              linear_q_agent,
              deep_q_agent]

    # Loop over the parameter set and create the Agents, Environments, and the Sim handler
    multi_sims = []
    for agt in agents:
        # Name the environment according to the agent used
        sim_ = sim.Sim(env=Small().build(),
                       tqdm_on=False,
                       agent=agt,
                       n_steps=steps)

        multi_sims.append(sim.MultiSim(sim_,
                                       n_jobs=1,  # Needs to be 1, DeepQAgent doesn't support pickle yet
                                       name='rl agents',
                                       n_reps=100))

    # Run all the sims. No need to parallelize here as it's done across n reps in MultiSim.run()
    for ms in tqdm(multi_sims):
        ms.run()

    fig = plot_dists(multi_sims, "Overall score")
    plt.show()
    fig.savefig('q_agent_comparison_score.png')

    fig = plot_dists(multi_sims, "Total deaths")
    plt.show()
    fig.savefig('q_agent_comparison_deaths.png')
