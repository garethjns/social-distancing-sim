from typing import Dict

from social_distancing_sim.agent.agent_base import AgentBase
from social_distancing_sim.environment.observation_space import ObservationSpace


class DummyAgent(AgentBase):
    """Doesn't do anything."""
    @property
    def available_actions(self) -> list:
        return []

    def _select_actions_targets(self) -> Dict[int, str]:
        return {}

