from social_distancing_sim.agent.agent_base import AgentBase
from typing import List, Dict

from social_distancing_sim.agent.agent_base import AgentBase
from social_distancing_sim.environment.observation_space import ObservationSpace


class TreatmentPolicyAgent(AgentBase):
    """
    TreatmentPolicyAgent applies treatment to random infected nodes in active time frame.

    It can be used to model start and end of social distancing or quarantine periods. Note that this agent is similar to
    the isolation agent, but will isolate any node not just infected ones.

    0         start['treat']            end['treat']
    |  Does nothing  |   Isolates ANY node    |   Does nothing ...

    """
    @property
    def available_actions(self) -> List[str]:
        return ['treat']

    @staticmethod
    def available_targets(obs: ObservationSpace) -> List[int]:
        """Slightly different IsolationAgent - also isolates clear nodes and reconnects any isolated node."""
        return obs.current_infected_nodes

    def select_actions(self, obs: ObservationSpace) -> Dict[int, str]:
        if len(self.currently_active_actions) > 0:
            # Don't track sample call here as self.get_actions() will handle that.
            return self.sample(obs,
                               track=False)
        else:
            return {}
