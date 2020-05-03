from typing import List, Dict

from social_distancing_sim.agent.agent_base import AgentBase


class VaccinationAgent(AgentBase):
    """VaccinationAgent randomly vaccinates clear nodes."""

    @property
    def available_actions(self) -> List[int]:
        """Isolation agent can only isolate. It can't even un-isolate (yet?)"""
        return [1]

    @property
    def available_targets(self) -> List[int]:
        return list(set(self.env.observation_space.current_clear_nodes).difference(
            self.env.observation_space.current_immune_nodes))

    def _select_actions_targets(self) -> Dict[int, int]:
        # Don't track sample call here as self.get_actions() will handle that.
        return self.sample(track=False)
