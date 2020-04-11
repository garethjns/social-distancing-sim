from typing import List

from social_distancing_sim.agent.agent import Agent
from social_distancing_sim.population.population import Population


class VaccinationAgent(Agent):
    def available_actions(self) -> List[str]:
        return ['vaccinate']

    def select_target(self, pop: Population) -> int:
        not_immune = set(pop.observation_space.current_clear_nodes).difference(
            pop.observation_space.current_immune_nodes)
        if len(not_immune) > 0:
            return self.state.choice(list(not_immune))

    def select_action(self, pop: Population) -> str:
        return self.available_actions()[0]
