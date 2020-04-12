from social_distancing_sim.agent.vaccination_agent import VaccinationAgent
from social_distancing_sim.disease.disease import Disease
from social_distancing_sim.environment.environment import Environment
from social_distancing_sim.environment.environment_plotting import EnvironmentPlotting
from social_distancing_sim.environment.graph import Graph
from social_distancing_sim.environment.healthcare import Healthcare
from social_distancing_sim.environment.observation_space import ObservationSpace
from social_distancing_sim.sim.sim import Sim

if __name__ == "__main__":
    seed = 123

    pop = Environment(name="agent example environment",
                      disease=Disease(name='COVID-19',
                                      virulence=0.01,
                                      seed=seed,
                                      immunity_mean=0.95,
                                      immunity_decay_mean=0.05),
                      healthcare=Healthcare(capacity=5),
                      observation_space=ObservationSpace(graph=Graph(community_n=15,
                                                                     community_size_mean=10,
                                                                     seed=seed + 1),
                                                         test_rate=1,
                                                         seed=seed + 2),
                      seed=seed + 3,
                      environment_plotting=EnvironmentPlotting(ts_fields_g2=["Score", "Action cost", "Overall score"],
                                                               ts_obs_fields_g2=["Observed Score", "Action cost",
                                                                                 "Observed overall score"]))

    sim = Sim(pop=pop,
              agent_delay=10,
              agent=VaccinationAgent(actions_per_turn=25,
                                     seed=seed),
              plot=True,
              save=True)

    sim.run()
