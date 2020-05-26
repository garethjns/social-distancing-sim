from typing import List, Dict

from social_distancing_sim.agent.agent_base import AgentBase


class VaccinationPolicyAgent(AgentBase):
    """
    Vaccination applies vaccination during a set time period.

    Unlike VaccinationAgent, vaccinates any clear node even if they have some immunity.

    It can be used to model availability of a vaccine, for max or staggered use.

     0         start['vaccinate']                 end['vaccinate']
    |   Does nothing   |   vaccinates ANY clear node      |      Does nothing ...
    """

    @property
    def available_actions(self) -> List[int]:
        return [1]

    @property
    def available_targets(self) -> List[int]:
        """Same as VaccinationAgent."""
        return list(self.env.observation_space.current_clear_nodes)

    def _select_actions_targets(self) -> Dict[int, int]:
        if len(self.currently_active_actions) > 0:
            # Don't track sample call here as self.get_actions() will handle that.
            return self.sample(track=False)
        else:
            return {}
