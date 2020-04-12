from joblib import Parallel, delayed

from social_distancing_sim.agent.isolation_agent import IsolationAgent
from social_distancing_sim.agent.vaccination_agent import VaccinationAgent
from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.environment.environment import Environment
from social_distancing_sim.environment.graph import Graph
from social_distancing_sim.environment.healthcare import Healthcare
from social_distancing_sim.environment.observation_space import ObservationSpace
from social_distancing_sim.sim.sim import Sim


def run_and_replay(sim):
    sim.run()
    if sim.save:
        sim.env.replay()


if __name__ == "__main__":
    seed = 123
    common_population_kwargs = {"disease": Disease(name='COVID-19',
                                                   virulence=0.0025,
                                                   seed=seed,
                                                   immunity_mean=0.99,
                                                   immunity_decay_mean=0.01),
                                "healthcare": Healthcare(capacity=200),
                                "observation_space": ObservationSpace(graph=Graph(community_n=50,
                                                                                  community_size_mean=15,
                                                                                  seed=seed + 1),
                                                                      test_rate=1,
                                                                      seed=seed + 2),
                                "initial_infections": 5,
                                "seed": seed + 3}

    sim_common_kwargs = {"n_steps": 130,
                         'plot': False,
                         'save': True,
                         'tqdm_on': True}

    sim_1 = Sim(Environment(name="Vaccination agent - early",
                            **common_population_kwargs),
                agent=VaccinationAgent(actions_per_turn=3),
                **sim_common_kwargs)

    sim_2 = Sim(Environment(name="Vaccination agent - late",
                            **common_population_kwargs),
                agent=VaccinationAgent(actions_per_turn=3),
                **sim_common_kwargs)

    sim_3 = Sim(Environment(name="Isolation agent - early",
                            **common_population_kwargs),
                agent=IsolationAgent(actions_per_turn=3),
                **sim_common_kwargs)

    sim_4 = Sim(Environment(name="Isolation agent - late",
                            **common_population_kwargs),
                agent=IsolationAgent(actions_per_turn=3),
                **sim_common_kwargs)

    Parallel(n_jobs=4,
             backend='loky')(delayed(run_and_replay)(sim) for sim in [sim_1, sim_2, sim_3, sim_4])
