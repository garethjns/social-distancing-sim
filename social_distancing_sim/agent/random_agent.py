from typing import Dict

from social_distancing_sim.agent.agent_base import AgentBase
from social_distancing_sim.environment.observation_space import ObservationSpace


class RandomAgent(AgentBase):
    def select_actions(self, obs: ObservationSpace,
                       n: int = 1) -> Dict[int, str]:
        n = self._check_available_targets(obs, n)

        return self.sample(obs, n)
