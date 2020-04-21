from dataclasses import dataclass
from typing import List, Union

import numpy as np


@dataclass
class ActionSpace:
    """
    Available actions and associated costs.

    TODO: Standardise api and remove **kwargs
    """
    vaccinate_cost: float = -2
    isolate_cost: float = 0
    treat_cost: float = -3
    vaccinate_efficiency: float = 0.95
    isolate_efficiency: float = 0.95
    reconnect_efficiency: float = 0.95
    treatment_conclusion_chance: float = 0.9
    treatment_recovery_rate_modifier: float = 1.5
    seed: Union[int, None] = None

    def __post_init__(self) -> None:
        self._prepare_random_state()

    def _prepare_random_state(self) -> None:
        self.state = np.random.RandomState(seed=self.seed)

    @property
    def n(self):
        return len(self.available_actions)

    @property
    def available_actions(self) -> List[str]:
        return ['vaccinate', 'isolate', 'reconnect', 'treat']

    def treat(self, **kwargs) -> float:
        kwargs["env"].disease.conclude(kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]],
                                       chance_to_force=self.treatment_conclusion_chance,
                                       recovery_rate_modifier=self.treatment_recovery_rate_modifier)
        kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]]["status"].immune = True
        kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]]["last_tested"] = kwargs["step"]

        return self.treat_cost

    def vaccinate(self, **kwargs) -> float:
        kwargs["env"].disease.give_immunity(kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]],
                                            immunity=self.vaccinate_efficiency)
        kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]]["status"].immune = True
        kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]]["last_tested"] = kwargs["step"]

        return self.vaccinate_cost

    def isolate(self, **kwargs) -> float:
        kwargs["env"].observation_space.graph.isolate_node(kwargs["target_node_id"],
                                                           effectiveness=self.isolate_efficiency)
        kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]]["status"].isolated = True

        return self.isolate_cost

    def reconnect(self, **kwargs) -> float:
        kwargs["env"].observation_space.graph.reconnect_node(kwargs["target_node_id"],
                                                             effectiveness=self.reconnect_efficiency)
        kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]]["status"].isolated = False

        return self.isolate_cost

    def sample(self):
        """Return a random available action"""
        return self.state.choice(self.available_actions)

    def clone(self) -> "ActionSpace":
        """Return a random available action"""
        return ActionSpace(seed=self.seed, vaccinate_cost=self.vaccinate_cost, isolate_cost=self.isolate_cost)
