from typing import List, Dict

from social_distancing_sim.agent.agent_base import AgentBase


class TreatmentPolicyAgent(AgentBase):
    """
    TreatmentPolicyAgent applies treatment to random infected nodes in active time frame.

    It can be used to model start and end of social distancing or quarantine periods. Note that this agent is similar to
    the isolation agent, but will isolate any node not just infected ones.

    0          start['treat']                    end['treat']
    |  Does nothing  |   treats ANY infected node    |   Does nothing ...

    """

    @property
    def available_actions(self) -> List[int]:
        return [4]

    @property
    def available_targets(self) -> List[int]:
        """Slightly different IsolationAgent - also isolates clear nodes and reconnects any isolated node."""
        return self.env.observation_space.current_infected_nodes

    def _select_actions_targets(self) -> Dict[int, int]:
        if len(self.currently_active_actions) > 0:
            # Don't track sample call here as self.get_actions() will handle that.
            return self.sample(track=False)
        else:
            return {}
