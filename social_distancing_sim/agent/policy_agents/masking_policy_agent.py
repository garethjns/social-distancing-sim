from typing import List, Dict

from social_distancing_sim.agent.agent_base import AgentBase


class MaskingPolicyAgent(AgentBase):
    """
    MaskingPolciyAgent hands out masks masks during a set time period.

    Any alive node can receive a mask. This includes those that already have masks. Masks currently don't decay (like
    immunity does) so this very crudely models their ongoing replacement.

     0         start['masking']                 end['masking']
    |   Does nothing   |   masks ANY alive node      |      Does nothing ...
    """

    @property
    def available_actions(self) -> List[int]:
        return [5]

    @property
    def available_targets(self) -> List[int]:
        return list(self.env.observation_space.current_alive_nodes)

    def _select_actions_targets(self) -> Dict[int, int]:
        if len(self.currently_active_actions) > 0:
            # Don't track sample call here as self.get_actions() will handle that.
            return self.sample(track=False)
        else:
            return {}
