from typing import List, Dict

from social_distancing_sim.agent.agent_base import AgentBase
from social_distancing_sim.environment.observation_space import ObservationSpace


class IsolationAgent(AgentBase):
    """Isolation agent will either isolate known infected, connected nodes, or reconnect known clear, isolated nodes."""
    @property
    def available_actions(self) -> List[str]:
        return ['isolate', 'reconnect']

    @staticmethod
    def available_targets(obs: ObservationSpace) -> Dict[str, List[int]]:
        return {'isolate': list(set(obs.current_infected_nodes).difference(obs.isolated_nodes)),
                'reconnect': list(set(obs.current_clear_nodes).intersection(obs.isolated_nodes))}

    def select_actions(self, obs: ObservationSpace) -> Dict[int, str]:
        actions = self._random_state.choice(self.available_actions,
                                            size=self.actions_per_turn)
        available_targets = self.available_targets(obs)

        available_actions = {}
        for ac in actions:
            available_targets_for_this_action = available_targets[ac]
            if len(available_targets_for_this_action) > 0:
                available_actions.update({self._random_state.choice(available_targets_for_this_action): ac})

        return available_actions
