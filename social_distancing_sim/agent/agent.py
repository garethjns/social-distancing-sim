import abc
import copy
from typing import List, Union, Dict, Tuple

import numpy as np

from social_distancing_sim.agent.action_space import ActionSpace
from social_distancing_sim.population.population import Population


class Agent(metaclass=abc.ABCMeta):
    action_space = ActionSpace()

    def __init__(self,
                 name: str = 'unnamed_agent',
                 seed: Union[None, int] = None,
                 actions_per_turn: int = 5) -> None:
        self.seed = seed
        self.name = name
        self.actions_per_turn = actions_per_turn
        self._prepare_random_state()

    def _prepare_random_state(self) -> None:
        self.state = np.random.RandomState(seed=self.seed)

    @abc.abstractmethod
    def available_actions(self) -> List[str]:
        pass

    @abc.abstractmethod
    def select_action(self, pop: Population) -> str:
        pass

    @abc.abstractmethod
    def select_target(self, pop: Population) -> int:
        pass

    def act(self, pop, **kwargs) -> Tuple[Dict[str, int], float]:

        completed_actions = {}
        action_cost = 0
        for _ in range(self.actions_per_turn):
            action = self.select_action(pop)
            target_node_id = self.select_target(pop)

            if (action is not None) and (target_node_id is not None):
                action_cost = getattr(self.action_space, action)(pop=pop,
                                                                 target_node_id=target_node_id,
                                                                 **kwargs)
                completed_actions.update({action: target_node_id})

        return completed_actions, action_cost

    def clone(self) -> "Agent":
        """Clone a fresh object with same seed (could be None)."""
        clone = copy.deepcopy(self)
        clone._prepare_random_state()
        return clone


