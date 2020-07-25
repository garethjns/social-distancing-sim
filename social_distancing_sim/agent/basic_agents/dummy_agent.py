from typing import Dict

from social_distancing_sim.agent.non_learning_agent_base import NonLearningAgentBase


class DummyAgent(NonLearningAgentBase):
    """Doesn't do anything."""
    @property
    def available_actions(self) -> list:
        return []

    def _select_actions_targets(self) -> Dict[int, str]:
        return {}

