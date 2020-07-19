from typing import List, Dict

from social_distancing_sim.agent.non_learning_agent_base import NonLearningAgentBase


class IsolationAgent(NonLearningAgentBase):
    """Isolation agent will either isolate known infected, connected nodes, or reconnect known clear, isolated nodes."""
    @property
    def available_actions(self) -> List[int]:
        return [2, 3]

    @property
    def available_targets(self) -> Dict[int, List[int]]:
        return {2: list(set(self.env.sds_env.observation_space.current_infected_nodes).difference
                        (self.env.sds_env.observation_space.current_isolated_nodes)),
                3: list(set(self.env.sds_env.observation_space.current_clear_nodes).intersection(
                    self.env.sds_env.observation_space.current_isolated_nodes))}

    def _select_actions_targets(self) -> Dict[int, str]:
        """Selects randomly between both actions, any time frames are totally ignored."""
        actions = self._random_state.choice(self.available_actions,
                                            replace=True,
                                            size=self.actions_per_turn)

        available_actions = {}
        for ac in actions:
            available_targets_for_this_action = self.available_targets[ac]
            if len(available_targets_for_this_action) > 0:
                available_actions.update({self._random_state.choice(available_targets_for_this_action): ac})

        return available_actions
