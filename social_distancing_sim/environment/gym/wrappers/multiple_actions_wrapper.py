from typing import List

import gym


class MultipleActionsWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, n_actions: int) -> None:
        super().__init__(env)
        self.n_actions = n_actions

    def action(self, action: int) -> List[int]:
        return [action] * self.n_actions

    def reverse_action(self, action):
        """Nope."""
        pass
