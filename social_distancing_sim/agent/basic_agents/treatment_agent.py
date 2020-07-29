from typing import List, Dict

from social_distancing_sim.agent.non_learning_agent_base import NonLearningAgentBase


class TreatmentAgent(NonLearningAgentBase):
    """TreatmentAgent randomly vaccinates clear nodes."""

    @property
    def available_actions(self) -> List[int]:
        """Isolation agent can only isolate. It can't even un-isolate (yet?)"""
        return [4]

    @property
    def available_targets(self) -> List[int]:
        return self.env.sds_env.observation_space.current_infected_nodes

    def _select_actions_targets(self) -> Dict[int, int]:
        # Don't track sample call here as self.get_actions() will handle that.
        return self.sample(track=False)
