from dataclasses import dataclass
from typing import List, Union, Dict

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
    reconnect_cost: float = 0
    mask_cost: float = -0.1
    vaccinate_efficiency: float = 0.95
    isolate_efficiency: float = 0.95
    reconnect_efficiency: float = 0.95
    treatment_conclusion_chance: float = 0.9
    treatment_recovery_rate_modifier: float = 1.5
    mask_efficiency: float = 0.25
    seed: Union[int, None] = None

    def __post_init__(self) -> None:
        self._prepare_random_state()

    def _prepare_random_state(self) -> None:
        self.state = np.random.RandomState(seed=self.seed)

    @property
    def n(self):
        return len(self.available_actions)

    @property
    def supported_actions(self) -> Dict[str, int]:
        return {'nothing': 0,
                'vaccinate': 1,
                'isolate': 2,
                'reconnect': 3,
                'treat': 4,
                'provide_mask': 5}

    @property
    def available_actions(self) -> List[int]:
        return list(self.supported_actions.values())

    @property
    def available_action_ids(self) -> List[int]:
        return list(self.supported_actions.values())

    def get_action_id(self, name: str) -> int:
        return self.supported_actions[name]

    def get_action_name(self, action_id: int) -> str:
        act = {v: k for k, v in self.supported_actions.items()}
        return act[action_id]

    def treat(self, **kwargs) -> float:
        kwargs["env"].disease.conclude(kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]],
                                       chance_to_force=self.treatment_conclusion_chance,
                                       recovery_rate_modifier=self.treatment_recovery_rate_modifier)
        kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]]["status"].immune = True
        kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]]["last_tested"] = kwargs["step"]

        return self.treat_cost

    def vaccinate(self, **kwargs) -> float:
        kwargs["env"].disease.give_immunity(
            kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]],
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

        return self.reconnect_cost

    def provide_mask(self, **kwargs) -> float:
        kwargs["env"].observation_space.graph.mask_node(kwargs["target_node_id"],
                                                        effectiveness=self.mask_efficiency)
        kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]]["status"].masked = True

        return self.mask_cost

    def remove_mask(self, **kwargs) -> float:
        kwargs["env"].observation_space.graph.unmask_node(kwargs["target_node_id"])
        kwargs["env"].observation_space.graph.g_.nodes[kwargs["target_node_id"]]["status"].masked = False

        return self.mask_cost

    def sample(self):
        """Return a random available action"""
        return self.state.choice(list(self.supported_actions.values()))

    def clone(self) -> "ActionSpace":
        """Return a random available action"""
        return ActionSpace(seed=self.seed, vaccinate_cost=self.vaccinate_cost, isolate_cost=self.isolate_cost)

    @classmethod
    def select_random_target(cls, n: int, available_targets: List[int]) -> List[int]:
        """Given a list of available targets, select a number of targets."""
        n_available = len(available_targets)
        valid = list(np.random.choice(available_targets,
                                      size=min(n, n_available),
                                      replace=False))
        diff = n - n_available
        invalid = [-1] * diff

        return valid + invalid
