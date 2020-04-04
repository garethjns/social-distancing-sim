from dataclasses import dataclass

from social_distancing_sim.agent.agent import Agent, IsolationAgent
from social_distancing_sim.population.population import Population


@dataclass
class Sim:
    pop: Population
    agent: Agent
    n_steps: int = 100
    plot: bool = False
    save: bool = False
    agent_delay: int = 5

    def run(self):

        for s in range(self.n_steps):

            self.pop.step(plot=self.plot,
                          save=self.save)

            if s > self.agent_delay:
                self.agent.act(self.pop)


if __name__ == "__main__":
    from social_distancing_sim.population.graph import Graph
    from social_distancing_sim.population.healthcare import Healthcare
    from social_distancing_sim.population.observation_space import ObservationSpace
    from social_distancing_sim.population.population import Population
    from social_distancing_sim.disease.disease import Disease

    seed = 123

    pop = Population(name="example population",
                     disease=Disease(name='COVID-19',
                                     virulence=0.02,
                                     seed=seed),
                     healthcare=Healthcare(capacity=5),
                     observation_space=ObservationSpace(graph=Graph(community_n=15,
                                                                    community_size_mean=10,
                                                                    seed=seed),
                                                        test_rate=1,
                                                        seed=seed),
                     seed=seed)

    sim = Sim(pop=pop,
              agent=IsolationAgent(actions_per_turn=5,
                                   seed=seed),
              plot=True,
              save=True)

    sim.run()
