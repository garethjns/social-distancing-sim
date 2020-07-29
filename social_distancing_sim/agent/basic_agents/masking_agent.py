from typing import List, Dict

from social_distancing_sim.agent.non_learning_agent_base import NonLearningAgentBase


class MaskingAgent(NonLearningAgentBase):
    """Provides masks to alive nodes. Masks modify virulence."""

    @property
    def available_actions(self) -> List[int]:
        return [5]

    @property
    def available_targets(self) -> List[int]:
        """Masks given out to any alive nodes """
        return self.env.sds_env.observation_space.current_alive_nodes

    def _select_actions_targets(self) -> Dict[int, int]:
        # Don't track sample call here as self.get_actions() will handle that.
        return self.sample(track=False)
