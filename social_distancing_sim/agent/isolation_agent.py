from typing import List

from social_distancing_sim.agent.agent import Agent
from social_distancing_sim.population.population import Population


class IsolationAgent(Agent):
    def available_actions(self) -> List[str]:
        return ['isolate']

    def select_target(self, pop: Population) -> int:
        infected_not_isolated = set(pop.observation_space.current_infected_nodes).difference(
            pop.observation_space.isolated_nodes)

        if len(infected_not_isolated):
            return self.state.choice(list(infected_not_isolated))

    def select_action(self, pop: Population) -> str:
        return self.available_actions()[0]
