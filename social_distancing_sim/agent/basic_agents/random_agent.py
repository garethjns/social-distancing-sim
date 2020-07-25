from typing import Dict, List

from social_distancing_sim.agent.non_learning_agent_base import NonLearningAgentBase


class RandomAgent(NonLearningAgentBase):
    """
    RandomAgent randomly selects an action and target.

    It doesn't support reconnection action, as this breaks on connected nodes - this is a good error check so will stay
    as it is for now.
    """

    @property
    def available_actions(self) -> List[int]:
        return [0, 1, 2, 4, 5, 6]

    def _select_actions_targets(self) -> Dict[int, int]:
        return self.sample()
