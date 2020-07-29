"""Run multiple simulations using some basic agents, some policy agents, and a trained linear q learner."""

from typing import List

import gym
import matplotlib.pyplot as plt
import seaborn as sns
from reinforcement_learning_keras.agents.components.helpers.virtual_gpu import VirtualGPU
from tqdm import tqdm

import social_distancing_sim.agent as agent
import social_distancing_sim.sim as sim
from scripts.visual_compare_multi_agents import DISTANCING_PARAMS, MASKING_PARAMS, TREATMENT_PARAMS, VACCINATION_PARAMS
from social_distancing_sim.agent.rl_agents.q_learning.dqn_untargeted import DQNUntargeted


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
    gpu = VirtualGPU(gpu_memory_limit=4096,
                     gpu_device_id=1)
    gym.envs.register(id='SDS-746-v0',
                      entry_point='social_distancing_sim.environment.gym.environments.sds_746:SDS746',
                      max_episode_steps=1000)
    env_spec = gym.make('SDS-746-v0').spec
    steps = 10

    deep_q_agent = DQNUntargeted.load('scripts/flat_obs_dqn_SDS-746-v0', actions_per_turn=5)

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
              deep_q_agent]
    agents = [agent.DummyAgent(name='Dummy agent'), deep_q_agent]

    # Loop over the parameter set and create the Agents, Environments, and the Sim handler
    multi_sims = []
    for agt in agents:
        sim_ = sim.Sim(env_spec=env_spec,
                       tqdm_on=False,
                       agent=agt,
                       n_steps=steps)

        multi_sims.append(sim.MultiSim(sim_,
                                       n_jobs=1,
                                       name='rl agents',
                                       n_reps=5))

    # Run all the sims. No need to parallelize here as it's done across n reps in MultiSim.run()
    for ms in tqdm(multi_sims):
        ms.run()

    fig = plot_dists(multi_sims, "Overall score")
    plt.show()
    fig.savefig('q_agent_comparison_score.png')

    fig = plot_dists(multi_sims, "Total deaths")
    plt.show()
    fig.savefig('q_agent_comparison_deaths.png')
