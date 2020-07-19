from typing import List, Dict

from social_distancing_sim.agent.non_learning_agent_base import NonLearningAgentBase


class VaccinationAgent(NonLearningAgentBase):
    """VaccinationAgent randomly vaccinates clear nodes."""

    @property
    def available_actions(self) -> List[int]:
        return [1]

    @property
    def available_targets(self) -> List[int]:
        return list(set(self.env.sds_env.observation_space.current_clear_nodes).difference(
            self.env.sds_env.observation_space.current_immune_nodes))

    def _select_actions_targets(self) -> Dict[int, int]:
        # Don't track sample call here as self.get_actions() will handle that.
        return self.sample(track=False)
