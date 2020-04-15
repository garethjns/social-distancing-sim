from typing import Dict, List

from social_distancing_sim.agent.agent_base import AgentBase
from social_distancing_sim.environment.observation_space import ObservationSpace


class RandomAgent(AgentBase):
    @property
    def available_actions(self) -> List[str]:
        return ['vaccinate', 'isolate']

    def select_actions(self, obs: ObservationSpace) -> Dict[int, str]:
        return self.sample(obs)
