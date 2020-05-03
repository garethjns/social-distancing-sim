from typing import List, Dict, Union

from social_distancing_sim.agent.agent_base import AgentBase
from social_distancing_sim.environment.environment import Environment


class MultiAgent(AgentBase):
    """
    Combine other agents/policy agents

    Actions per turn is dynamic and determined by individual agents
    """

    def __init__(self, agents: List[AgentBase], env: Union[Environment, None] = None, *args, **kwargs):
        self.agents = agents
        self.actions_per_turn = sum([a.actions_per_turn for a in agents])
        self.set_env(env)

        super().__init__(env, *args, **kwargs)

    def set_env(self, env: Union[None, Environment]) -> None:
        self.env = env
        for agt in self.agents:
            agt.set_env(self.env)

    def _select_actions_targets(self) -> Dict[int, int]:
        """Ask each agent for their actions. They handle n and availability"""

        actions = []
        targets = []
        for agent in self.agents:
            acts, tars = agent.get_actions()
            actions.extend(acts)
            targets.extend(tars)

        return {t: a for t, a in zip(targets, actions)}
