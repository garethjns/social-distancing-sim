import copy
from dataclasses import dataclass
from typing import Union, Dict, Hashable, Any

import numpy as np


@dataclass
class Disease:
    """

    :param immunity_mean: Mean prop. immunity gained after survival.
    :param immunity_std:
    :param immunity_decay_mean: Mean prop. immunity decay per step.
    """
    seed: Union[None, int] = None
    name: str = 'COVID-19'
    virulence: float = 0.006
    recovery_rate: float = 0.98
    duration_mean: float = 21
    duration_std: float = 5

    immunity_mean: float = 0.8
    immunity_std: float = 0.02
    immunity_decay_mean: float = 0.1
    immunity_decay_std: float = 0.005

    def __post_init__(self) -> None:
        self._prepare_random_state()

    def _prepare_random_state(self) -> None:
        self.state = np.random.RandomState(seed=self.seed)

    def give_immunity(self, node: Dict[Hashable, Any],
                      immunity: float = None) -> Dict[Hashable, Any]:
        """
        Give immunity to a node.

        :param node: Graph node data.
        :param immunity: Amount of immunity to give to node, optional. Default None, which gives self.immunity_ (the
                         values applied on recovery). Allows for bonus immunity from actions, like vaccination.
        :return: Modified node data.
        """
        if immunity is None:
            immunity = min(self.immunity_mean + self.state.normal(scale=self.immunity_std), 1.0)
        node["immune"] = immunity

        return node

    def decay_immunity(self, node: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        """
        Decay immunity on a node.

        Uses self.immunity_decay_ values. Not option to specify decay amount for now, because YAGNI.

        :param node: Graph node data.
        :return: Modified node data.
        """
        decay = self.immunity_decay_mean + self.state.normal(scale=self.immunity_decay_std)
        new_immunity = max(0.0, node["immune"] - node["immune"] * decay)
        node["immune"] = new_immunity
        return node

    def modified_virulence(self, immunity: float) -> float:
        """Reduce virulence according to immunity"""
        return min(max(1e-7, self.virulence * (1 - immunity)), 0.999)

    def conclude(self, node: Dict[Hashable, Any],
                 chance_to_force: float = 0.0,
                 recovery_rate_modifier: float = 1) -> Dict[Hashable, Any]:
        """

        :param node: Graph node to update.

        :param chance_to_force: Chance to force conclusion. Must be between 0 -> 1. Optional, default 0.
                                Default uses self.duration_* params instead. Allows for specifying a treatment efficacy
                                along with recovery rate modified. Note that if conclusion is forced, node can still
                                die. So treatment can be worse than cure if high chance to force conclusion and
                                recovery rate penalty.
        :param recovery_rate_modifier: Modify recovery rate depending on external factors such as healthcare burden.
                                       Default 1.0 (no modification). Only relevant if disease ends.

        :return: Updated graph node.
        """

        # Decide if forcing conclusion or not
        force = False
        if chance_to_force > 0:
            force = self.state.binomial(1, chance_to_force)

        # Decide end of disease
        if node["infected"] > self.state.normal(self.duration_mean,
                                                self.duration_std,
                                                size=1) or force:
            # Concluding, decide fate
            node["infected"] = 0
            modified_recovery_rate = max(0.0, min(1.0,  self.recovery_rate * recovery_rate_modifier))

            if self.state.binomial(1, modified_recovery_rate):
                node = self.give_immunity(node)
                node["alive"] = True
            else:
                node["alive"] = False

        else:
            # Continue disease progression
            node["infected"] += 1

        return node

    def try_to_infect(self, node: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        if not node.get("infected", 0) > 0:
            node["infected"] = self.state.binomial(1, self.modified_virulence(node.get("immune", 0)))
        return node

    @staticmethod
    def force_infect(node: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        node['infected'] = 1
        return node

    def clone(self):
        clone = copy.deepcopy(self)
        clone._prepare_random_state()
        return clone
