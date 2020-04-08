import abc
import copy
from typing import List, Callable, Union, Dict, Tuple

import numpy as np

from social_distancing_sim.agent.action_space import ActionSpace
from social_distancing_sim.population.population import Population


class Agent(metaclass=abc.ABCMeta):
    action_space = ActionSpace()

    def __init__(self,
                 name: str='unnamed_agent',
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


class IsolationAgent(Agent):
    def available_actions(self) -> List[str]:
        return ['isolate']

    def select_target(self, pop: Population) -> int:
        infected_not_isolated = set(pop.observation_space.current_infected_nodes).difference(
            pop.observation_space.isolated_nodes)
        if len(infected_not_isolated):
            return self.state.choice(list(infected_not_isolated))

    def select_action(self, pop: Population) -> str:
        return self.available_actions()[0]


class VaccinationAgent(Agent):
    def available_actions(self) -> List[str]:
        return ['vaccinate']

    def select_target(self, pop: Population) -> int:
        not_immune = set(pop.observation_space.current_clear_nodes).difference(
            pop.observation_space.current_immune_nodes)
        if len(not_immune) > 0:
            return self.state.choice(list(not_immune))

    def select_action(self, pop: Population) -> str:
        return self.available_actions()[0]


class IsoVaccAgent:
    isolation_agent = IsolationAgent()
    vaccination_agent = VaccinationAgent()

    def available_actions(self) -> List[Callable]:
        return self.isolation_agent.available_actions() + self.vaccination_agent.available_actions()

    def select_target(self):
        pass

    def select_action(self):
        pass
