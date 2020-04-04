import abc
from typing import List, Callable, Union, Hashable, Dict, Any

import numpy as np

from social_distancing_sim.agent.action_space import ActionSpace
from social_distancing_sim.population.population import Population


class Agent(metaclass=abc.ABCMeta):
    _action_space = ActionSpace

    def __init__(self,
                 seed: Union[None, int] = None,
                 actions_per_turn: int = 5) -> None:
        self.seed = seed
        self.actions_per_turn = actions_per_turn
        self._prepare_random_state()

    def _prepare_random_state(self) -> None:
        self.state = np.random.RandomState(seed=self.seed)

    @abc.abstractmethod
    def available_actions(self) -> List[Callable]:
        pass

    @abc.abstractmethod
    def select_action(self, pop: Population) -> Callable:
        pass

    @abc.abstractmethod
    def select_target(self, pop: Population) -> int:
        pass

    def act(self, pop, **kwargs):
        for _ in range(self.actions_per_turn):
            action = self.select_action(pop)
            target_node_id = self.select_target(pop)

            if (action is not None) and (target_node_id is not None):
                action(pop=pop,
                       target_node_id=target_node_id,
                       **kwargs)


class IsolationAgent(Agent):
    def available_actions(self) -> List[Callable]:
        return [self._action_space.isolate]

    def select_target(self, pop: Population) -> int:
        infected_not_isolated = set(pop.observation_space.known_current_infected_nodes).difference(
             pop.observation_space.known_isolated_nodes)
        if len(infected_not_isolated):
            return self.state.choice(list(infected_not_isolated))

    def select_action(self, pop: Population) -> Callable:
        return self.available_actions()[0]
