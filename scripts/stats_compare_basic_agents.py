import numpy as np
from tqdm import tqdm

from social_distancing_sim.agent.isolation_agent import IsolationAgent
from social_distancing_sim.agent.vaccination_agent import VaccinationAgent
from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.population.graph import Graph
from social_distancing_sim.population.healthcare import Healthcare
from social_distancing_sim.population.observation_space import ObservationSpace
from social_distancing_sim.population.population import Population
from social_distancing_sim.sim.multi_sim import MultiSim
from social_distancing_sim.sim.sim import Sim

if __name__ == "__main__":
    seed = None
    pop = Population(name="example population",
                     disease=Disease(name='COVID-19',
                                     virulence=0.01,
                                     seed=seed,
                                     immunity_mean=0.95,
                                     immunity_decay_mean=0.05),
                     healthcare=Healthcare(capacity=5),
                     observation_space=ObservationSpace(graph=Graph(community_n=15,
                                                                    community_size_mean=10,
                                                                    seed=seed),
                                                        test_rate=1,
                                                        seed=seed),
                     seed=seed,
                     plot_ts_fields_g2=["Score", "Action cost", "Overall score"],
                     plot_ts_obs_fields_g2=["Observed Score", "Action cost", "Observed overall score"])

    multi_sims = []
    for n_actions, agent_delay, agent in np.array(np.meshgrid([5, 10, 15, 20, 25, 30], [5, 10, 20, 40, 80],
                                                              [VaccinationAgent, IsolationAgent])).T.reshape(-1, 3):
        multi_sims.append(MultiSim(Sim(pop=pop,
                                       n_steps=150,
                                       agent_delay=agent_delay,
                                       agent=agent(actions_per_turn=n_actions,
                                                   seed=seed)),
                                   name='basic agent comparison',
                                   n_reps=1000))

    for ms in tqdm(multi_sims):
        ms.run()
