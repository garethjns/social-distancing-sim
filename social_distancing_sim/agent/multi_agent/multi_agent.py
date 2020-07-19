from typing import List, Dict

from social_distancing_sim.agent.non_learning_agent_base import NonLearningAgentBase


class MultiAgent(NonLearningAgentBase):
    """
    Combine other agents/policy agents

    Actions per turn is dynamic and determined by individual agents
    """

    def __init__(self, agents: List[NonLearningAgentBase], env_spec: str,
                 *args, **kwargs):
        self.agents = agents
        self.actions_per_turn = sum([a.actions_per_turn for a in agents])
        super().__init__(env_spec, *args, **kwargs)
        self.set_child_envs()

    def set_child_envs(self) -> None:
        """Parent agent is built by env_builder from spec, child agents need to be attached."""

        for agt in self.agents:
            agt.attach_to_env(self.env)

    def _select_actions_targets(self) -> Dict[int, int]:
        """Ask each agent for their actions. They handle n and availability"""

        actions = []
        targets = []
        for agent in self.agents:
            acts, tars = agent.get_actions()
            actions.extend(acts)
            targets.extend(tars)

        return {t: a for t, a in zip(targets, actions)}
