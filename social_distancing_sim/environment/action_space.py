from dataclasses import dataclass
from typing import List, Union

import numpy as np


@dataclass
class ActionSpace:
    """
    Available actions and associated costs.

    TODO: Standardise api and remove **kwargs
    """
    vaccinate_cost: int = -2
    isolate_cost: int = 0
    seed: Union[int, None] = None

    def __post_init__(self) -> None:
        self._prepare_random_state()

    def _prepare_random_state(self) -> None:
        self.state = np.random.RandomState(seed=self.seed)

    @property
    def available_actions(self) -> List[str]:
        return ['vaccinate', 'isolate', 'reconnect']

    def vaccinate(self, **kwargs) -> int:
        kwargs["env"].disease.give_immunity(kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]])
        kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]]["status"].immune = True
        kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]]["last_tested"] = kwargs["step"]

        return self.vaccinate_cost

    def isolate(self, **kwargs) -> int:
        kwargs["env"].observation_space.graph.isolate_node(kwargs["target_node_id"])
        kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]]["status"].isolated = True

        return self.isolate_cost

    def reconnect(self, **kwargs) -> int:
        kwargs["env"].observation_space.graph.reconnect_node(kwargs["target_node_id"])
        kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]]["status"].isolated = False

        return self.isolate_cost

    def sample(self):
        """Return a random available action"""
        return self.state.choice(self.available_actions)

    def clone(self) -> "ActionSpace":
        """Return a random available action"""
        return ActionSpace(seed=self.seed, vaccinate_cost=self.vaccinate_cost, isolate_cost=self.isolate_cost)
