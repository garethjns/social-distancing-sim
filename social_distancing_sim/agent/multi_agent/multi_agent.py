from typing import List, Dict

from social_distancing_sim.agent.agent_base import AgentBase
from social_distancing_sim.environment.observation_space import ObservationSpace


class MultiAgent(AgentBase):
    """
    Combine other agents/policy agents

    Actions per turn is dynamic and determined by individual agents
    """

    def __init__(self, agents: List[AgentBase], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.agents = agents
        self.actions_per_turn = sum([a.actions_per_turn for a in agents])

    def select_actions(self, obs: ObservationSpace) -> Dict[int, str]:
        """Ask each agent for their actions. They handle n and availability"""

        actions = {}
        for agent in self.agents:
            actions.update(agent.get_actions(obs))

        return actions
    