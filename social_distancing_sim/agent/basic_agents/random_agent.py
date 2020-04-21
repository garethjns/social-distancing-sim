from typing import Dict, List

from social_distancing_sim.agent.agent_base import AgentBase
from social_distancing_sim.environment.observation_space import ObservationSpace


class RandomAgent(AgentBase):
    """
    RandomAgent randomly selects an action and target.

    It doesn't support reconnection action, as this breaks on connected nodes - this is a good error check so will stay
    as it is for now.
    """
    @property
    def available_actions(self) -> List[str]:
        return ['vaccinate', 'isolate']

    def select_actions(self, obs: ObservationSpace) -> Dict[int, str]:
        return self.sample(obs)
