from typing import List, Dict

from social_distancing_sim.agent.agent_base import AgentBase


class DistancingPolicyAgent(AgentBase):
    """
    DistancingPolicyAgent applies isolation on a set turn and reconnection on a later turn.

    It can be used to model start and end of social distancing or quarantine periods. Note that this agent is similar to
    the isolation agent, but will isolate any node not just infected ones.

    0         start['isolate']         end['isolate']         start['reconnect']         end['reconnect']
    |  Does nothing  |   Isolates ANY node    |       Does nothing      |   Reconnects Nodes   |  Does nothing ...

    """

    @property
    def available_actions(self) -> List[int]:
        return [2, 3]

    @property
    def available_targets(self) -> Dict[int, List[int]]:
        """Slightly different IsolationAgent - also isolates clear nodes and reconnects any isolated node."""
        return {2: list(set(self.env.observation_space.current_clear_nodes).difference(
            self.env.observation_space.current_isolated_nodes)),
            3: self.env.observation_space.current_isolated_nodes}

    def _select_actions_targets(self) -> Dict[int, str]:
        """Selects from actions that are currently available. If both are active, selects randomly between them."""

        available_actions = {}
        if len(self.currently_active_actions) > 0:
            actions = self._random_state.choice(self.currently_active_actions,
                                                replace=True,
                                                size=self.actions_per_turn)
            available_targets = self.available_targets

            # This effectively discards duplicate actions/target if same target is randomly selected twice
            # Not sure if this is a good approach. When pool of targets is small, more likely to not take all available
            # actions.
            for ac in actions:
                available_targets_for_this_action = available_targets[ac]
                if len(available_targets_for_this_action) > 0:
                    available_actions.update({self._random_state.choice(available_targets_for_this_action): ac})

        return available_actions
